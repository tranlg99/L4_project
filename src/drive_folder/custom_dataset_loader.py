from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class TaiChiDataset(Dataset):
    """Tai Chi dataset."""
    def __init__(self,log_file,root_dir,check=False,transform=None):
        """
        Args: 
		log_file (string): path to txt file with all logged sample ids. 
		root_dir (string): Directory with all the image frames.
        check (Bool): Also return 7th frames for sanity check.
		transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.sample_names=open(log_file).read().splitlines()
        self.root_dir=root_dir
        self.check = check
        self.transform=transform
    def __len__(self):
        return len(self.sample_names)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
            sample_id = self.sample_names[idx]
            image0_path=os.path.join(self.root_dir,"frame0",sample_id)
            coords_path=os.path.join(self.root_dir,"coords",sample_id)
            image0=np.load(image0_path)
            coords= np.load(coords_path)

            if self.check:
                image1_path=os.path.join(self.root_dir,"frame1",sample_id)
                image1=np.load(image1_path)
                sample={'image0':image0, 'image1':image1, 'coords':coords}
            else:
                sample={'image0':image0, 'coords':coords}
        if self.transform:
            sample= self.transform(sample)
        return sample

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    - swap color axis because
    - numpy image: H x W x C
    - torch image: C x H x W
    """
    def __call__(self, sample):
        image0, coords = sample['image0'], sample['coords']
        image0 = image0.transpose((2, 0, 1))

        if len(sample)==3:
            image1 = sample['image1']
            image1 = image1.transpose((2, 0, 1))
            return {'image0': torch.from_numpy(image0),
                    'image1': torch.from_numpy(image1),
                    'coords': torch.from_numpy(coords)}
        else:
            return {'image0': torch.from_numpy(image0),
                    'coords': torch.from_numpy(coords)}