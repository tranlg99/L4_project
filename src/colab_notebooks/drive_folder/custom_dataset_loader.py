import albumentations as A
import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def get_dataloader(dataset, batch_size=16, shuffle=True):
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
  return dataloader

class CustomDataset(Dataset):
    """Our custom dataset."""
    def __init__(self,log_file,root_dir,check=False,transform=None):
        """
        Args: 
		    log_file (string): path to txt file with all logged sample ids. 
		    root_dir (string): Directory with all the image frames.
            check (Bool, optional): Also return 7th frames for sanity check.
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
        image3_path=os.path.join(self.root_dir,"frame3",sample_id)
        coords_path=os.path.join(self.root_dir,"coords",sample_id)
        vis_path=os.path.join(self.root_dir,"vis",sample_id)
	
        image0=np.load(image0_path+'.npy')
        image3=np.load(image3_path+'.npy')
        image0 = cv2.resize(image0, dsize=(640, 360))
        image3 = cv2.resize(image3, dsize=(640, 360))


        coords= np.load(coords_path+'.npy')
        vis=np.load(vis_path+'.npy')
        

        if self.check:
            image7_path=os.path.join(self.root_dir,"frame7",sample_id)
            image7=np.load(image7_path+'.npy')
            image7 = cv2.resize(image7, dsize=(640, 360))
            sample={'id':sample_id, 'image0':image0, 'image3':image3, 'image7':image7, 'coords':coords, 'vis':vis, 'shift':np.array((0,0))}
        else:
          sample={'id':sample_id, 'image0':image0, 'image3':image3, 'coords':coords, 'vis':vis, 'shift':np.array((0,0))}
             
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    - swap color axis because
    - numpy image: H x W x C
    - torch image: C x H x W
    """
    def __call__(self, sample):
        sample_id, image0, image3, coords, vis, shift = sample['id'], sample['image0'], sample['image3'], sample['coords'], sample['vis'], sample['shift']
        image0 = image0.transpose((2, 0, 1))
        image3 = image3.transpose((2, 0, 1))


        if len(sample)==7:
            image7 = sample['image7']
            image7 = image7.transpose((2, 0, 1))
            return {'id': sample_id,
                    'image0': torch.from_numpy(image0),
                    'image3': torch.from_numpy(image3),
                    'image7': torch.from_numpy(image7),
                    'coords': torch.from_numpy(coords),
		                'vis': torch.from_numpy(vis),
                    'shift': torch.from_numpy(shift)}
        else:
            return {'id': sample_id,
                    'image0': torch.from_numpy(image0),
                    'image3': torch.from_numpy(image3),
                    'coords': torch.from_numpy(coords),
		                'vis': torch.from_numpy(vis),
                    'shift': torch.from_numpy(shift)}

class AugmentData(object):
    """
    Augment data with ColorJitter, Gaussian Noise and To Gray transformations.
    """
    def __call__(self, sample):
        sample_id, image0, image3, coords, vis, shift = sample['id'], sample['image0'], sample['image3'], sample['coords'], sample['vis'], sample['shift']
        
        trans = A.Compose(
            [
             A.augmentations.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.2),
             A.augmentations.transforms.GaussNoise (var_limit=.05, mean=0, per_channel=True, always_apply=False, p=0.2),
             A.augmentations.transforms.ToGray(p=0.1)
             ],
            additional_targets={'image3': 'image'}
            )

        image0 = image0.astype(np.float32)
        image3 = image3.astype(np.float32)

        transformed = trans(image=image0, image3=image3)
        n_image0 = transformed['image']
        n_image3 = transformed['image3']
        
        if len(sample)==7:
            image7 = sample['image7']
            return {'id': sample_id,
                    'image0': n_image0,
                    'image3': n_image3,
                    'image7': image7,
                    'coords': coords,
		            'vis': vis,
                    'shift': shift}
        else:
            return {'id': sample_id,
                    'image0': n_image0,
                    'image3': n_image3,
                    'coords': coords,
		            'vis': vis,
                    'shift': shift}

class ShiftData(object):
    """
    Shifting data along random x, y axis.
    """
    def __call__(self, sample):
        sample_id, image0, image3, coords, vis, shift = sample['id'], sample['image0'], sample['image3'], sample['coords'], sample['vis'], sample['shift']

        # randomly sample these values and pass to Affine trans (x in range (-5,5), y in range (-10,10)
        random_x = np.random.randint(-5,5)
        random_y = np.random.randint(-10,10)

        coords[:, :, 0] =  coords[:, :, 0 ]+random_y
        coords[:, :, 1] =  coords[:, :, 1 ]+random_x


        trans = A.Compose(
            [
              A.augmentations.geometric.transforms.Affine(scale=1, translate_percent=None, translate_px={'x':random_x, 'y':random_y}, shear=None, fit_output=False, keep_ratio=False, always_apply=False, p=1)             
            ],
            additional_targets={'image3': 'image'}
            )

        image0 = image0.astype(np.float32)
        image3 = image3.astype(np.float32)

        transformed = trans(image=image0, image3=image3)
        n_image0 = transformed['image']
        n_image3 = transformed['image3']

        shift = (random_x, random_y)
 
        if len(sample)==7:
            image7 = sample['image7']
            return {'id': sample_id,
                    'image0': n_image0,
                    'image3': n_image3,
                    'image7': image7,
                    'coords': coords,
		            'vis': vis,
                    'shift': np.array(shift)}
        else:
            return {'id': sample_id,
                    'image0': n_image0,
                    'image3': n_image3,
                    'coords': coords,
		            'vis': vis,
                    'shift': np.array(shift)}