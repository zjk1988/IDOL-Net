import h5py 
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat,savemat
import torch.utils.data as Data
class myDataset(Dataset):
    def __init__(self,file_path1,file_path2,file_path3,file_path4,file_path5,file_path6,file_path7):
        
        if not os.path.isdir(file_path1):
            raise ValueError("input file_path is not a dir")
        self.file_path1 = file_path1
        self.gt_CT = os.listdir(file_path1)
        self.gt_CT.sort()
        
        if not os.path.isdir(file_path2):
            raise ValueError("input file_path is not a dir")
        self.file_path2 = file_path2
        self.Xma = os.listdir(file_path2)
        self.Xma.sort()
        
        if not os.path.isdir(file_path3):
            raise ValueError("input file_path is not a dir")
        self.file_path3 = file_path3
        self.metal_trace = os.listdir(file_path3)
        self.metal_trace.sort()

        if not os.path.isdir(file_path4):
            raise ValueError("input file_path is not a dir")
        self.file_path4 = file_path4
        self.ma_proj = os.listdir(file_path4)
        self.ma_proj.sort()     

        if not os.path.isdir(file_path5):
            raise ValueError("input file_path is not a dir")
        self.file_path5 = file_path5
        self.LI_proj = os.listdir(file_path5)
        self.LI_proj.sort() 
        
        
        if not os.path.isdir(file_path6):
            raise ValueError("input file_path is not a dir")
        self.file_path6 = file_path6
        self.mask = os.listdir(file_path6)
        self.mask.sort() 


        if not os.path.isdir(file_path7):
            raise ValueError("input file_path is not a dir")
        self.file_path7 = file_path7
        self.gt_proj = os.listdir(file_path7)
        self.gt_proj.sort() 


        
    def __getitem__(self, index):
        

        f = os.path.join(self.file_path1, self.gt_CT[index])
        data = h5py.File(f,'r')
        data = data['gt_CT'][:].T
        data = np.expand_dims(data, axis=0)
        gt_CT = torch.tensor(data)
    
        f = os.path.join(self.file_path2, self.Xma[index])
        data = h5py.File(f,'r')
        data = data['ma_CT'][:].T
        data = np.expand_dims(data, axis=0)
        Xma = torch.tensor(data)   
        
        f = os.path.join(self.file_path4, self.ma_proj[index])
        data = loadmat(f)
        data = data['ma_proj']
        data = np.expand_dims(data, axis=0)
        S_ma = torch.tensor(data)        

        
        f = os.path.join(self.file_path3, self.metal_trace[index])
        data = h5py.File(f,'r')
        data = data['metal_trace'][:]
        data = np.expand_dims(data, axis=0)
        mt = torch.tensor(data)
        
      
        f = os.path.join(self.file_path5, self.LI_proj[index])
        data = loadmat(f)
        data = data['LI_proj']
        data = np.expand_dims(data, axis=0)
        S_LI = torch.tensor(data)
      
        f = os.path.join(self.file_path6, self.mask[index])
        data = h5py.File(f,'r')
        data = data['metal_im'][:].T
        data = np.expand_dims(data, axis=0)
        mask = torch.tensor(data)
        
        f = os.path.join(self.file_path7, self.gt_proj[index])
        data = data = loadmat(f)
        data = data['gt_proj']
        data = np.expand_dims(data, axis=0)
        gt_proj = torch.tensor(data.astype(np.float32))
        

        return S_ma, S_LI, mt, Xma, mask, gt_proj, gt_CT

    def __len__(self):
        return 90000

# file_path1 = "../gt_CT"
# file_path2 = '../ma_CT2021/'
# file_path3 = '../metal_trace/'
# file_path4 = '../test_data/ma_proj_new'
# file_path5 = '../test_data/LI_proj_new'
# file_path6 = '../mask_im/'
# file_path7 = '../test_data/gt_proj_new'

train_dataset = myDataset(file_path1,file_path2,file_path3,file_path4,file_path5,file_path6,file_path7)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        num_workers=12,
                        pin_memory=True,
                        sampler=train_sampler)