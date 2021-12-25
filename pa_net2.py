# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 08:56:42 2020

@author: szm
"""
print('assddd2')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# from test1 import *
import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import os
import time
import h5py
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import transforms
from scipy.io import loadmat,savemat
import torch.utils.data as Data
from PIL import Image
# import torchvision.transforms as T
import torch.utils.data 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator
import torch.distributed as dist
import argparse
import odl
from odl.contrib import torch as odl_torch
import functools
from copy import deepcopy,copy
from blocks import ConvolutionBlock,ResidualBlock,FullyConnectedBlock
from torch.nn import init

print('a111')
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--random_sample', action='store_true', default=True, help='whether to sample the dataset with random sampler')
args = parser.parse_args()

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)

num_epoch = 400
lr = 0.0001

batch_size = 1
print('batch_size',batch_size)
file_path1 = "../gt_CT"
file_path2 = '../ma_CT2021/'
file_path3 = '../metal_trace/'
file_path4 = '../test_data/ma_proj_new'
file_path5 = '../test_data/LI_proj_new'
file_path6 = '../mask_im/'
file_path7 = '../test_data/gt_proj_new'
 
if args.local_rank==0:
    print('my')    


class Encoder(nn.Module):
    def __init__(self, input_ch, base_ch, num_down, num_residual, res_norm='instance', down_norm='instance'):
        super(Encoder, self).__init__()

        self.conv0 = ConvolutionBlock(
            in_channels=input_ch, out_channels=base_ch, kernel_size=3, stride=1,
            padding=1, pad='reflect', norm=down_norm, activ='relu')
        
        output_ch = base_ch
        for i in range(1, num_down+1):
            m = ConvolutionBlock(
                in_channels=output_ch, out_channels=output_ch * 2, kernel_size=4,
                stride=2, padding=1, pad='reflect', norm=down_norm, activ='relu')
            setattr(self, "conv{}".format(i), m)
            output_ch *= 2

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                ResidualBlock(output_ch, pad='reflect', norm=res_norm, activ='relu'))

        self.layers = [getattr(self, "conv{}".format(i)) for i in range(num_down+1)] + \
            [getattr(self, "res{}".format(i)) for i in range(num_residual)]
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, output_ch, base_ch, num_up, num_residual, num_sides, size, res_norm='instance', up_norm='layer', fuse=False):
        super(Decoder, self).__init__()
        input_ch = base_ch * 2 ** num_up
        input_chs = []

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                ResidualBlock(input_ch, pad='reflect', norm=res_norm, activ='lrelu'))
            input_chs.append(input_ch)

        for i in range(num_up-1):
            m = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvolutionBlock(
                    in_channels=input_ch, out_channels=input_ch // 2, kernel_size=5,
                    stride=1, padding=2, pad='reflect', norm=up_norm, activ='lrelu'))
            setattr(self, "conv{}".format(i), m)
            input_chs.append(input_ch)
            input_ch //= 2

        m = nn.Sequential(
            nn.Upsample(size=size, mode="nearest"),
            ConvolutionBlock(
                in_channels=input_ch, out_channels=input_ch // 2, kernel_size=5,
                stride=1, padding=2, pad='reflect', norm=up_norm, activ='lrelu'))
        setattr(self, "conv{}".format(num_up-1), m)
        input_chs.append(input_ch)
        input_ch //= 2
         
        m = ConvolutionBlock(
            in_channels=base_ch, out_channels=output_ch, kernel_size=7,
            stride=1, padding=3, pad='reflect', norm='none', activ='tanh')
        setattr(self, "conv{}".format(num_up), m)
        input_chs.append(base_ch)
        
        self.layers = [getattr(self, "res{}".format(i)) for i in range(num_residual)] + \
            [getattr(self, "conv{}".format(i)) for i in range(num_up + 1)]

        # If true, fuse (concat and conv) the side features with decoder features
        # Otherwise, directly add artifact feature with decoder features
        if fuse:
            input_chs = input_chs[-num_sides:]
            for i in range(num_sides):
                setattr(self, "fuse{}".format(i),
                    nn.Conv2d(input_chs[i] * 2, input_chs[i], 1))
            self.fuse = lambda x, y, i: getattr(self, "fuse{}".format(i))(torch.cat((x, y), 1))
        else:
            self.fuse = lambda x, y, i: x + y

    def forward(self, x):
        m = len(self.layers)
        for i in range(m):
            x = self.layers[i](x)
        return x



class Projection(nn.Module):
    def __init__(self):
        super(Projection,self).__init__()
        reco_space = odl.uniform_discr(
        min_pt=[-127, -127], max_pt=[128, 128], shape=[256, 256], dtype='float32')

        # Angles: uniformly spaced, n = 1000, min = 0, max = pi
        angle_partition = odl.uniform_partition(0, np.pi, 361)
        
        # Detector: uniformly sampled, n = 500, min = -30, max = 30
        detector_partition = odl.uniform_partition(-183, 183, 367)
        
        # Make a parallel beam geometry with flat detector
        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)    
        # Ray transform (= forward projection).
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
        
        self.proj = odl_torch.OperatorModule(ray_trafo)
#        self.recon = odl_torch.OperatorModule(fbp) 
    def forward(self,x):
        return self.proj(x)


class FBP(nn.Module):
    def __init__(self):
        super(FBP,self).__init__()
        reco_space = odl.uniform_discr(
        min_pt=[-127, -127], max_pt=[128, 128], shape=[256, 256], dtype='float32')

        # Angles: uniformly spaced, n = 1000, min = 0, max = pi
        angle_partition = odl.uniform_partition(0, np.pi, 361)
        
        # Detector: uniformly sampled, n = 500, min = -30, max = 30
        detector_partition = odl.uniform_partition(-183, 183, 367)
        
        # Make a parallel beam geometry with flat detector
        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)    
        # Ray transform (= forward projection).
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry,impl='astra_cuda')
        
        # Fourier transform in detector direction
        fourier = odl.trafos.FourierTransform(ray_trafo.range, axes=[1])
        
        # Create ramp in the detector direction
        ramp_function = fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))
        
        # Create ramp filter via the convolution formula with fourier transforms
        ramp_filter = fourier.inverse * ramp_function * fourier
        
        # Create filtered back-projection by composing the back-projection (adjoint)
        # with the ramp filter.
        fbp = ray_trafo.adjoint * ramp_filter

        self.recon = odl_torch.OperatorModule(fbp) 
        
    def forward(self,x):
        return self.recon(x)




class Fusion(nn.Module):
    def __init__(self):
        super(Fusion,self).__init__()
        self.fuse = nn.Conv2d(2,1,kernel_size=1,stride=1,padding=0)
    def forward(self,x1,x2):
        return self.fuse(torch.cat((x1,x2),dim=1))

class FirstBlock(nn.Module):
    """
    Image with artifact is denoted as low quality image
    Image without artifact is denoted as high quality image
    """

    def __init__(self, input_ch=1, base_ch=32, num_down=2, num_residual=2, num_sides="all",
        res_norm='instance', down_norm='instance', up_norm='layer', fuse=True, shared_decoder=False):
        super(FirstBlock, self).__init__()
        
        self.n = num_down + num_residual + 1 if num_sides == "all" else num_sides
        self.encoder_sino1 = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.encoder_sino2 = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.encoder_im1 = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.encoder_im2 = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)

        self.decoder_sino1 = Decoder(input_ch, base_ch, num_down, num_residual, self.n, (361,367), res_norm, up_norm, fuse)
        self.decoder_sino2 = Decoder(input_ch, base_ch, num_down, num_residual, self.n, (361,367), res_norm, up_norm, fuse)
        self.decoder_im1 = Decoder(input_ch, base_ch, num_down, num_residual, self.n,  (256,256), res_norm, up_norm, fuse)
        self.decoder_im2 = Decoder(input_ch, base_ch, num_down, num_residual, self.n,  (256,256), res_norm, up_norm, fuse)
        
        self.proj = Projection()
        self.fbp = FBP()
        # self.fuse1 = Fusion()
        # self.fuse2 = Fusion()


    def forward(self, S_ma, S_LI, mt, Xma):

        feature_en_sino1 = self.encoder_sino1(S_ma).cuda(args.local_rank)
        feature_en_sino2 = self.encoder_sino2(S_ma).cuda(args.local_rank)
        clean_sino =  self.decoder_sino1(feature_en_sino1).cuda(args.local_rank)
        noise_sino = self.decoder_sino2(feature_en_sino2).cuda(args.local_rank)

        feature_en_im1 = self.encoder_im1(Xma).cuda(args.local_rank)
        feature_en_im2 = self.encoder_im2(Xma).cuda(args.local_rank)
        clean_im = self.decoder_im1(feature_en_im1).cuda(args.local_rank)
        noise_im = self.decoder_im2(feature_en_im2).cuda(args.local_rank)

        sino1 = self.proj(clean_im)
        X1 = self.fbp(clean_sino)

        # re_sino = self.fuse1(clean_sino,sino1)
        # re_im = self.fuse2(clean_im,X1)



        return clean_sino, noise_sino, clean_im, noise_im, sino1, X1
class Block(nn.Module):
    def __init__(self):
        super(Block,self).__init__()
        self.senet = SE_net()
        self.imnet = IM_net()
        self.proj = Projection()
        self.fbp = FBP()
        # self.fuse1 = Fusion()
        # self.fuse2 = Fusion()
    def forward(self,sino1,sino2,SLI,mt,Xin1,Xin2):

        sub1 = (sino1-SLI).mul(mt)
        sub2 = (sino2-SLI).mul(mt)
        sub = torch.cat((sub1,sub2),dim=1)
        sino = self.senet(sub,mt).mul(mt)+SLI.mul(1-mt)
        X = self.imnet(torch.cat((Xin1,Xin2),dim=1),Xin1)

        sino1 = self.proj(X)
        X1 = self.fbp(sino)

        # re_sino = self.fuse1(sino,sino1)
        # re_im = self.fuse2(X,X1)
        return sino, X, sino1, X1

class BlockLast(nn.Module):
    def __init__(self):
        super(BlockLast,self).__init__()
        self.senet = SE_net()
        self.imnet = IM_net()
        self.fbp = FBP()
        self.fuse = Fusion()
    def forward(self,sino1,sino2,SLI,mt,Xin1,Xin2):
        sub1 = (sino1-SLI).mul(mt)
        sub2 = (sino2-SLI).mul(mt)
        sub = torch.cat((sub1,sub2),dim=1)
        sino = self.senet(sub,mt).mul(mt)+SLI.mul(1-mt)
        X = self.imnet(torch.cat((Xin1,Xin2),dim=1),Xin1)
        X1 = self.fbp(sino)
        re_im = self.fuse(X,X1)
        return re_im


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.firstB = FirstBlock()
        self.block2 = Block()
        self.block3 = Block()
        self.lastB = BlockLast()

    def forward(self,S_ma, S_LI, mt, Xma):
        out1 = self.firstB(S_ma, S_LI, mt, Xma)
        out2 = self.block2(out1[0],out1[4],S_LI,mt,out1[2],out1[5])
        out3 = self.block3(out2[0],out2[2],S_LI,mt,out2[1],out2[3])
        out4 = self.lastB(out3[0],out3[2],S_LI,mt,out3[1],out3[3])

        return out1[0], out1[1], out1[2], out1[3], out1[4], out1[5], out4




train_dataset = myDataset(file_path1,file_path2,file_path3,file_path4,file_path5,file_path6,file_path7)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        num_workers=12,
                        pin_memory=True,
                        sampler=train_sampler)


net = Net()
save_path11 = './models'
pre_dict = torch.load(os.path.join(save_path11,'10parallNet-.pth'))
new_pre = {}
for k,v in pre_dict.items():
   name = k[7:]
   new_pre[name] = v

net.load_state_dict(new_pre)

net.cuda(args.local_rank)
net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[args.local_rank],find_unused_parameters=True)

save_path1 = './models'
criterion1 = nn.L1Loss().cuda(args.local_rank)
criterion2 = nn.L1Loss().cuda(args.local_rank)
criterion3 = nn.L1Loss().cuda(args.local_rank)
criterion4 = nn.L1Loss().cuda(args.local_rank)
criterion5 = nn.L1Loss().cuda(args.local_rank)
criterion6 = nn.L1Loss().cuda(args.local_rank)
criterion7 = nn.L1Loss().cuda(args.local_rank)
criterion8 = nn.L1Loss().cuda(args.local_rank)
criterion9 = nn.L1Loss().cuda(args.local_rank)
class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
    
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                # if k != 'meta':
                self.batch[k] = self.batch[k].cuda(args.local_rank)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

opt = torch.optim.Adam(net.parameters(),lr=lr,betas=(0.5, 0.999))
StepLR = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)

