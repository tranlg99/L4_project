import torchvision
import numpy as np
import torch
import argparse
import cv2
from PIL import Image
from google.colab.patches import cv2_imshow
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim
import time
import copy
import torchvision.transforms as transforms
from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import matplotlib.pyplot as plt
from random import sample

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def meshgrid2d(B, Y, X, stack=False, device='cuda'):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x

def get_grid(N, B, H, W, normalise=False):
  N_ = np.sqrt(N).round().astype(np.int32)
  grid_y, grid_x = meshgrid2d(B, N_, N_, stack=False, device='cuda')
  grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
  grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)

  if normalise:
    # normalise to values of range [-1, 1] - x = -1, y = -1 is the left-top pixel
    grid_x = (grid_x - W) / W 
    grid_y = (grid_y - H) / H

  xy = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2
  xy = xy.view(B, N_, N_, 2)

  return xy

import matplotlib.pyplot as plt

def plot_learning_curve(train_losses, val_losses):
  train_epoch_list = [x[0] for x in train_losses]
  val_epoch_list = [x[0] for x in val_losses]

  t_losses = [x[1] for x in train_losses]
  v_losses = [x[1] for x in val_losses]

  plt.plot(train_epoch_list, t_losses, label = "train loss")
  plt.plot(val_epoch_list, v_losses, label = "valid loss")
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend()
  plt.show()
  


def split_dataset(dataset, validation_split, batch_size, shuffle_dataset, random_seed):
  dataset_size = len(dataset)
  indices = list(range(dataset_size))
  split = int(np.floor(validation_split * dataset_size))
  if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
  train_indices, val_indices = indices[split:], indices[:split]
  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)
  
  train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
  validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

  return train_loader, validation_loader

def initialise_model(device, to_learn=[]):
  # Initialize model with the best available weights
  # create an instance of (e.g.) torchvision.models.segmentation.fcn_resnet50
  # and tell it to load pretrained weights
  weights = FCN_ResNet50_Weights.DEFAULT
  modified_model = fcn_resnet50(weights=weights)
  # print(modified_model)

  # Get first layer weights and double
  model_weights = {k: (v, v.dtype, v.shape) for k, v in modified_model.state_dict().items()}
  first_layer_weights = model_weights['backbone.conv1.weight'][0]
  # print('backbone.conv1.weight: {} {}'.format(model_weights['backbone.conv1.weight'][1], model_weights['backbone.conv1.weight'][2]))
  doubled_weights = first_layer_weights.repeat(1, 2, 1, 1)


  # modify the model by doubling input channel from 3 to 6 
  modified_model.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  # copy the doubled weights into first layer
  with torch.no_grad():
      modified_model.backbone.conv1.weight = torch.nn.Parameter(doubled_weights)

  # removing final layer and replacing with a 2D vector output at each pixel and visibility (instead of 20 class logits)
  # instead of torch.Size([1, 21, 120, 240]) -> torch.Size([1, 3, 120, 240])
  modified_model.classifier[3] = nn.Sequential()
  modified_model.classifier[4] = nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1))

  # choose which params to tune
  if "all" not in to_learn:
    for name, param in modified_model.named_parameters():
      if name not in to_learn:
        param.requires_grad = False

  # model to train() and load onto computation devicce
  modified_model.to(device)
  # print(modified_model)

  return modified_model

import numpy as np
import torch
import torch.nn as nn

def meshgrid2d(device, B, Y, X):
    """
    Returns a 2d meshgrid sized B x Y x X
    """
    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    return grid_y, grid_x

def get_grid(device, N, B, H, W, shift=None, normalise=False):
    """
    Get a point grid with equal spacing with N points
    Args:
        device (string): device to load resulting tensor to
        N (integer): number of points in the grid
        B (integer): batch size
        H, W (integer): height and width of pixel space of the grid
        shift (B, tuple(x,y), optional): xy shift for the whole grid to move in pixel space
        normalise (Bool, optional): normalise to -1, 1, if torch.nn.functional.grid_sample is used
    """

    if shift==None:
        shift=torch.zeros(B, 2)
        shift = shift.to(device)

    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = meshgrid2d(device, B, N_, N_)
    grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16) + shift[:,0].reshape(B,1).tile(N).reshape(B,N)
    grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16) + shift[:,1].reshape(B,1).tile(N).reshape(B,N)

    if normalise:
        # normalise to values of range [-1, 1] - x = -1, y = -1 is the left-top pixel
        grid_x = (grid_x - (W/2)) / (W/2)
        grid_y = (grid_y - (H/2)) / (H/2)

    xy = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2
    xy = xy.view(B, N_, N_, 2)

    return xy

def take_grid_points(device, coords, batch_size, outputs, shift):
    """
    Return grid point values from the full output using torch.nn.functional.grid_sample
    """
    # Take the values of outputs from the NxN grid points
    grid = get_grid(device, coords.shape[2], batch_size, outputs.shape[2], outputs.shape[3], shift=shift, normalise=True) # ([B, H(64), W(64), 2])
    outputs = torch.nn.functional.grid_sample(outputs, grid, align_corners=True) # torch.size([B,2,H,W])

    # Reshape to match target shape
    outputs = torch.permute(outputs, (0, 2, 3, 1)) # torch.Size([B, H, W, 2])

    return outputs

def get_predictions(device, outputs, coords, batch_size, shift=None):
    """
    Process models predictions for loss calculation
    """
    # Take relevant points from the grid
    outputs = take_grid_points(device, coords, batch_size, outputs, shift)

    # Get coords predictions
    outputs_coords = outputs[:, :, :, :2]
    # Coordinates = predicted displacement + grid points in (B, 360, 640, 2)
    original_points = get_grid(device, coords.shape[2], batch_size, 360, 640, shift=shift)
    outputs_coords = original_points + outputs_coords
    outputs_coords = outputs_coords.view(batch_size,1,-1,2) # torch.Size([B, 1, 64*64, 2])

    # Get vis predictions
    outputs_vis = outputs[:, : , :, -1] # changing to tensor with 0s and 1s??
    outputs_vis = outputs_vis.view(batch_size,1,-1) # torch.Size([B, 1, 64*64])

    return outputs_coords, outputs_vis

def zero_out_coords(coords, outputs_coords, vis):
    """
    Returns zeroed out coords in both ground truth and predictions so they dont contribute to coords loss
    """
    # Take points that are not visible from vis ground truth data
    point_visiblity = vis > 0 

    # Zero out those in both outputs_coords and coords
    outputs_coords_zeroed = torch.zeros_like(outputs_coords)
    coords_zeroed = torch.zeros_like(coords)

    outputs_coords_zeroed[:, :, :, 0] = torch.where(point_visiblity, outputs_coords[:, :, :, 0], 0)
    outputs_coords_zeroed[:, :, :, 1] = torch.where(point_visiblity, outputs_coords[:, :, :, 1], 0)
    coords_zeroed[:, :, :, 0] = torch.where(point_visiblity, coords[:, :, :, 0], 0)
    coords_zeroed[:, :, :, 1] = torch.where(point_visiblity, coords[:, :, :, 1], 0)

    return coords_zeroed, outputs_coords_zeroed
