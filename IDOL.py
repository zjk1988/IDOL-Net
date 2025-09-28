import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
import h5py
import numpy as np
from scipy.io import loadmat
import argparse
import odl
from odl.contrib import torch as odl_torch
from pathlib import Path
import logging
from typing import Tuple, Dict, Any

# Import your custom blocks
from blocks import ConvolutionBlock, ResidualBlock, FullyConnectedBlock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CT Metal Artifact Reduction Training')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true', default=True, 
                       help='whether to sample the dataset with random sampler')
    parser.add_argument('--num_epochs', default=400, type=int, help='number of training epochs')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size per GPU')
    parser.add_argument('--num_workers', default=12, type=int, help='number of data loading workers')
    parser.add_argument('--save_interval', default=10, type=int, help='save model every N epochs')
    parser.add_argument('--log_interval', default=100, type=int, help='log every N iterations')
    return parser.parse_args()


class ProjectionOperator(nn.Module):
    """ODL-based projection operator for CT reconstruction."""
    
    def __init__(self, image_shape: Tuple[int, int] = (256, 256), 
                 num_angles: int = 361, num_detectors: int = 367):
        super().__init__()
        
        # Define reconstruction space
        reco_space = odl.uniform_discr(
            min_pt=[-127, -127], max_pt=[128, 128], 
            shape=image_shape, dtype='float32'
        )
        
        # Define geometry
        angle_partition = odl.uniform_partition(0, np.pi, num_angles)
        detector_partition = odl.uniform_partition(-183, 183, num_detectors)
        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
        
        # Create projection operator
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
        self.proj = odl_torch.OperatorModule(ray_trafo)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FBPOperator(nn.Module):
    """ODL-based filtered backprojection operator."""
    
    def __init__(self, image_shape: Tuple[int, int] = (256, 256),
                 num_angles: int = 361, num_detectors: int = 367):
        super().__init__()
        
        # Define reconstruction space (same as projection)
        reco_space = odl.uniform_discr(
            min_pt=[-127, -127], max_pt=[128, 128], 
            shape=image_shape, dtype='float32'
        )
        
        # Define geometry
        angle_partition = odl.uniform_partition(0, np.pi, num_angles)
        detector_partition = odl.uniform_partition(-183, 183, num_detectors)
        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
        
        # Create operators
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
        
        # Create ramp filter
        fourier = odl.trafos.FourierTransform(ray_trafo.range, axes=[1])
        ramp_function = fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))
        ramp_filter = fourier.inverse * ramp_function * fourier
        
        # Create FBP operator
        fbp = ray_trafo.adjoint * ramp_filter
        self.recon = odl_torch.OperatorModule(fbp)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.recon(x)


class FeatureFusion(nn.Module):
    """Simple feature fusion module."""
    
    def __init__(self, in_channels: int = 2, out_channels: int = 1):
        super().__init__()
        self.fuse = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat((x1, x2), dim=1))


class FirstStageBlock(nn.Module):
    """First stage processing block with dual-domain reconstruction."""
    
    def __init__(self, input_ch: int = 1, base_ch: int = 32, num_down: int = 2, 
                 num_residual: int = 4, num_sides: str = "all",
                 res_norm: str = 'instance', down_norm: str = 'instance', 
                 up_norm: str = 'layer', fuse: bool = True):
        super().__init__()
        
        self.n = num_down + num_residual + 1 if num_sides == "all" else num_sides
        
        # Encoders for sinogram and image domains
        self.encoder_sino1 = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.encoder_sino2 = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.encoder_im1 = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.encoder_im2 = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        
        # Decoders
        self.decoder_sino1 = Decoder(input_ch, base_ch, num_down, num_residual, self.n, 
                                   (361, 367), res_norm, up_norm, fuse)
        self.decoder_sino2 = Decoder(input_ch, base_ch, num_down, num_residual, self.n, 
                                   (361, 367), res_norm, up_norm, fuse)
        self.decoder_im1 = Decoder(input_ch, base_ch, num_down, num_residual, self.n, 
                                 (256, 256), res_norm, up_norm, fuse)
        self.decoder_im2 = Decoder(input_ch, base_ch, num_down, num_residual, self.n, 
                                 (256, 256), res_norm, up_norm, fuse)
        
        # Projection and reconstruction operators
        self.proj = ProjectionOperator()
        self.fbp = FBPOperator()

    def forward(self, S_ma: torch.Tensor, S_LI: torch.Tensor, 
                mt: torch.Tensor, Xma: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        device = S_ma.device
        
        # Compute residual sinogram
        res = S_ma - S_LI
        
        # Process sinogram domain
        feature_en_sino1 = self.encoder_sino1(res)
        feature_en_sino2 = self.encoder_sino2(res)
        clean_sino = self.decoder_sino1(feature_en_sino1).mul(mt) + S_LI
        noise_sino = self.decoder_sino2(feature_en_sino2).mul(mt)

        # Process image domain
        feature_en_im1 = self.encoder_im1(Xma)
        feature_en_im2 = self.encoder_im2(Xma)
        clean_im = self.decoder_im1(feature_en_im1)
        noise_im = self.decoder_im2(feature_en_im2)

        # Cross-domain consistency
        sino1 = self.proj(clean_im)
        X1 = self.fbp(clean_sino)

        return clean_sino, noise_sino, clean_im, noise_im, sino1, X1


class ProcessingBlock(nn.Module):
    """Intermediate processing block."""
    
    def __init__(self):
        super().__init__()
        self.senet = SE_net()
        self.imnet = IM_net()
        self.proj = ProjectionOperator()
        self.fbp = FBPOperator()
        
    def forward(self, sino1: torch.Tensor, sino2: torch.Tensor, 
                SLI: torch.Tensor, mt: torch.Tensor, 
                Xin1: torch.Tensor, Xin2: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        
        # Process sinogram differences
        sub1 = (sino1 - SLI).mul(mt)
        sub2 = (sino2 - SLI).mul(mt)
        sub = torch.cat((sub1, sub2), dim=1)
        sino = self.senet(sub, mt).mul(mt) + SLI.mul(1 - mt)
        
        # Process images
        X = self.imnet(torch.cat((Xin1, Xin2), dim=1), Xin1)
        
        # Cross-domain operations
        sino1 = self.proj(X)
        X1 = self.fbp(sino)

        return sino, X, sino1, X1


class FinalBlock(nn.Module):
    """Final processing block with fusion."""
    
    def __init__(self):
        super().__init__()
        self.senet = SE_net()
        self.imnet = IM_net()
        self.fbp = FBPOperator()
        self.fuse = FeatureFusion()
        
    def forward(self, sino1: torch.Tensor, sino2: torch.Tensor,
                SLI: torch.Tensor, mt: torch.Tensor,
                Xin1: torch.Tensor, Xin2: torch.Tensor) -> torch.Tensor:
        
        sub1 = (sino1 - SLI).mul(mt)
        sub2 = (sino2 - SLI).mul(mt)
        sub = torch.cat((sub1, sub2), dim=1)
        sino = self.senet(sub, mt).mul(mt) + SLI.mul(1 - mt)
        
        X = self.imnet(torch.cat((Xin1, Xin2), dim=1), Xin1)
        X1 = self.fbp(sino)
        
        return self.fuse(X, X1)


class ReconstructionNet(nn.Module):
    """Main reconstruction network."""
    
    def __init__(self):
        super().__init__()
        self.first_block = FirstStageBlock()
        self.block2 = ProcessingBlock()
        self.block3 = ProcessingBlock()
        self.last_block = FinalBlock()

    def forward(self, S_ma: torch.Tensor, S_LI: torch.Tensor, 
                mt: torch.Tensor, Xma: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        
        # First stage
        out1 = self.first_block(S_ma, S_LI, mt, Xma)
        
        # Intermediate stages
        out2 = self.block2(out1[0], out1[4], S_LI, mt, out1[2], out1[5])
        out3 = self.block3(out2[0], out2[2], S_LI, mt, out2[1], out2[3])
        
        # Final stage
        out4 = self.last_block(out3[0], out3[2], S_LI, mt, out3[1], out3[3])

        return (*out1, out4)


class CTDataset(Dataset):
    """Optimized dataset class with better error handling and caching."""
    
    def __init__(self, data_paths: Dict[str, str]):
        self.data_paths = data_paths
        self._validate_paths()
        self._load_file_lists()
        
    def _validate_paths(self):
        """Validate all data paths exist."""
        for name, path in self.data_paths.items():
            if not Path(path).is_dir():
                raise ValueError(f"Path {name} ({path}) does not exist or is not a directory")
                
    def _load_file_lists(self):
        """Load and sort file lists."""
        self.file_lists = {}
        for name, path in self.data_paths.items():
            files = sorted(os.listdir(path))
            self.file_lists[name] = files
            logger.info(f"Found {len(files)} files in {name}")
            
        # Verify all lists have same length
        lengths = [len(files) for files in self.file_lists.values()]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError("All data directories must contain the same number of files")
            
    def _load_h5_data(self, filepath: str, key: str) -> np.ndarray:
        """Load data from HDF5 file."""
        try:
            with h5py.File(filepath, 'r') as f:
                data = f[key][:]
                if key == 'gt_CT':
                    data = data.T
            return np.expand_dims(data, axis=0).astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading H5 file {filepath}: {e}")
            raise
            
    def _load_mat_data(self, filepath: str, key: str) -> np.ndarray:
        """Load data from MAT file."""
        try:
            data = loadmat(filepath)[key]
            return np.expand_dims(data, axis=0).astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading MAT file {filepath}: {e}")
            raise

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        try:
            # Load ground truth CT
            gt_ct_path = Path(self.data_paths['gt_CT']) / self.file_lists['gt_CT'][index]
            gt_CT = torch.from_numpy(self._load_h5_data(gt_ct_path, 'gt_CT'))
            
            # Load metal artifact CT
            xma_path = Path(self.data_paths['ma_CT']) / self.file_lists['ma_CT'][index]
            Xma = torch.from_numpy(self._load_mat_data(xma_path, 'ma_CT'))
            
            # Load projections
            ma_proj_path = Path(self.data_paths['ma_proj']) / self.file_lists['ma_proj'][index]
            S_ma = torch.from_numpy(self._load_mat_data(ma_proj_path, 'ma_proj'))
            
            li_proj_path = Path(self.data_paths['LI_proj']) / self.file_lists['LI_proj'][index]
            S_LI = torch.from_numpy(self._load_mat_data(li_proj_path, 'LI_proj'))
            
            gt_proj_path = Path(self.data_paths['gt_proj']) / self.file_lists['gt_proj'][index]
            gt_proj = torch.from_numpy(self._load_mat_data(gt_proj_path, 'gt_proj'))
            
            # Load metal trace and mask
            mt_path = Path(self.data_paths['metal_trace']) / self.file_lists['metal_trace'][index]
            mt = torch.from_numpy(self._load_h5_data(mt_path, 'metal_trace'))
            
            mask_path = Path(self.data_paths['mask']) / self.file_lists['mask'][index]
            mask = torch.from_numpy(self._load_h5_data(mask_path, 'metal_im'))
            
            return S_ma, S_LI, mt, Xma, mask, gt_proj, gt_CT
            
        except Exception as e:
            logger.error(f"Error loading data at index {index}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.file_lists['gt_CT'])


class LossCalculator:
    """Centralized loss calculation."""
    
    def __init__(self, device):
        self.device = device
        self.l1_loss = nn.L1Loss()
        
    def compute_losses(self, outputs: Tuple[torch.Tensor, ...], 
                      targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        S_ma, S_LI, mt, Xma, mask, gt_proj, gt_CT = targets.values()
        
        # Unpack outputs
        clean_sino, noise_sino, clean_im, noise_im, sino1, X1, final_im = outputs
        
        # Ground truth components
        gt_sino_noise = S_ma - gt_proj
        gt_im_noise = Xma - gt_CT
        
        # Compute individual losses
        losses = {
            'clean_sino': self.l1_loss(clean_sino.mul(mt), gt_proj.mul(mt)) * 0.06,
            'noise_sino': self.l1_loss(noise_sino.mul(mt), gt_sino_noise.mul(mt)) * 0.06,
            'clean_im': self.l1_loss(clean_im, gt_CT),
            'noise_im': self.l1_loss(noise_im, gt_im_noise),
            'sino_consistency': self.l1_loss((clean_sino + noise_sino).mul(mt), S_ma.mul(mt)) * 0.06,
            'im_consistency': self.l1_loss((clean_im + noise_im), Xma),
            'cross_sino': self.l1_loss(sino1, gt_proj) * 0.06,
            'cross_im': self.l1_loss(X1, gt_CT),
            'final': self.l1_loss(final_im.mul(1 - mask), gt_CT.mul(1 - mask)) * 10
        }
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class Trainer:
    """Training orchestrator with better organization."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.local_rank}')
        self.setup_distributed()
        self.setup_model()
        self.setup_data()
        self.setup_optimization()
        self.loss_calculator = LossCalculator(self.device)
        
    def setup_distributed(self):
        """Initialize distributed training."""
        if self.args.local_rank != -1:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.args.local_rank)
            
    def setup_model(self):
        """Initialize model and load checkpoint if available."""
        self.model = ReconstructionNet()
        
        # Load pretrained weights if available
        checkpoint_path = Path('./models/17parallNet2021.pth')
        if checkpoint_path.exists():
            self.load_checkpoint(checkpoint_path)
            
        self.model.to(self.device)
        
        if self.args.local_rank != -1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.args.local_rank],
                find_unused_parameters=True
            )
            
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            # Remove 'module.' prefix if present
            state_dict = {}
            for k, v in checkpoint.items():
                name = k[7:] if k.startswith('module.') else k
                state_dict[name] = v
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded checkpoint from {path}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            
    def setup_data(self):
        """Setup data loaders."""
        data_paths = {
            'gt_CT': "../gt_CT",
            'ma_CT': '../ma_CT2021/',
            'metal_trace': '../metal_trace/',
            'ma_proj': '../test_data/ma_proj_new',
            'LI_proj': '../test_data/LI_proj_new',
            'mask': '../mask_im/',
            'gt_proj': '../test_data/gt_proj_new'
        }
        
        dataset = CTDataset(data_paths)
        
        if self.args.local_rank != -1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = None
            
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=True
        )
        
        self.sampler = sampler
        
    def setup_optimization(self):
        """Setup optimizer and scheduler."""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(0.5, 0.999)
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,
            gamma=0.5
        )
        
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        if self.sampler:
            self.sampler.set_epoch(epoch)
            
        self.model.train()
        epoch_losses = []
        
        for i, batch in enumerate(self.data_loader):
            # Move data to device
            batch = [x.to(self.device, non_blocking=True) for x in batch]
            S_ma, S_LI, mt, Xma, mask, gt_proj, gt_CT = batch
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(S_ma, S_LI, mt, Xma)
            
            # Compute losses
            targets = {
                'S_ma': S_ma, 'S_LI': S_LI, 'mt': mt, 'Xma': Xma,
                'mask': mask, 'gt_proj': gt_proj, 'gt_CT': gt_CT
            }
            losses = self.loss_calculator.compute_losses(outputs, targets)
            
            # Backward pass
            losses['total'].backward()
            self.optimizer.step()
            
            epoch_losses.append(losses['total'].item())
            
            # Logging
            if i % self.args.log_interval == 0 and self.args.local_rank <= 0:
                logger.info(f'Epoch {epoch}, Iter {i}, Loss: {losses["total"].item():.6f}')
                
        return np.mean(epoch_losses)
        
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        if self.args.local_rank <= 0:
            checkpoint_path = Path('./models') / f'checkpoint_epoch_{epoch}.pth'
            checkpoint_path.parent.mkdir(exist_ok=True)
            
            state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
            torch.save(state_dict, checkpoint_path)
            logger.info(f'Saved checkpoint to {checkpoint_path}')
            
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.args.num_epochs} epochs")
        
        for epoch in range(27, self.args.num_epochs):  # Resume from epoch 27
            epoch_loss = self.train_epoch(epoch)
            self.scheduler.step()
            
            if self.args.local_rank <= 0:
                logger.info(f'Epoch {epoch} completed, Average Loss: {epoch_loss:.6f}')
                
            if epoch % self.args.save_interval == 0:
                self.save_checkpoint(epoch)


def main():
    """Main function."""
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()