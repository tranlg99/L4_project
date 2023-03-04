from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import torch.nn as nn
import torch
import numpy as np
import time
from datetime import datetime
import copy
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from .checkpoints import save_checkpoint


def initialise_model(device, to_learn=[], drop_out=0):
  # Initialize model with the best available weights
  # create an instance of (e.g.) torchvision.models.segmentation.fcn_resnet50
  # and tell it to load pretrained weights
  weights = FCN_ResNet50_Weights.DEFAULT
  modified_model = fcn_resnet50(weights=weights)
  # print(modified_model)

  # Get first layer weights and double
  model_weights = {k: (v, v.dtype, v.shape) for k, v in modified_model.state_dict().items()}
  first_layer_weights = model_weights['backbone.conv1.weight'][0]
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

  # add drop out layers, last layer if 1, and second layer if 2
  if drop_out==1:
    modified_model.classifier[3] = nn.Dropout(p=0.2)
  if drop_out == 2:
    modified_model.classifier[3] = nn.Dropout(p=0.2)
    modified_model.classifier[0] = nn.Sequential(nn.Dropout(p=0.2), nn.Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))

  # choose which params to tune
  if "all" not in to_learn:
    for name, param in modified_model.named_parameters():
      if name not in to_learn:
        param.requires_grad = False

  # model to train() and load onto computation devicce
  modified_model.to(device)
  print("Model initialised")

  return modified_model

def train_model(device, model, dataloaders, optimizer, past_t_loss, past_v_loss, best_weights =[], num_epochs = 100, verbose = True, clip = 1.0, eval_freq=5, description="", path=""):
  model.to(device)
  since = time.time()

  best_model_wts = best_weights
  past_v_losses = np.array([x[1] for x in past_v_loss])
  if len(past_v_losses)!=0:
    best_loss = np.min(past_v_losses)
  else:
    best_loss = 100

  # Log loss history
  train_loss_history = past_t_loss
  val_loss_history = past_v_loss
  global_epoch = len(past_t_loss)

  # Set up loss
  coords_criterion = nn.MSELoss()
  vis_criterion = nn.BCELoss()

  print("Starting training: {} epochs".format(num_epochs))

  train_size = len(dataloaders['train'].dataset)
  valid_size = len(dataloaders['val'].dataset)

  if valid_size==0:
    phases = ['train']
  else:
    phases = ['train', 'val']

  for epoch in range(0,num_epochs):
    for phase in phases:
      # Set model to training mode, if it is time for validation set to eval mode
      if phase == 'train':
        model.train() 
      elif phase == 'val' and (epoch%eval_freq==0 or epoch == num_epochs-1):
        model.eval()
      else:
        continue

      running_loss = 0.0
      running_vis_loss = 0.0
      running_coords_loss = 0.0
      total_samples = 0

      # Iterate over data.
      for i_batch, sample_batched in enumerate(dataloaders[phase]):
        batch_size = len(sample_batched['id'])
        total_samples+=batch_size

        # Get input images and concatenate image tensors on channel dimension
        input1 = sample_batched['image0']
        input2 = sample_batched['image3']
        inputs = torch.cat((input1, input2), dim=1) 

        # Get ground truth coords and vis
        coords = sample_batched['coords']
        vis = torch.where(sample_batched['vis'] > 0, 1.0, 0.0)
        shift = sample_batched['shift']

        inputs = inputs.to(device).float() # torch.Size([B, 6, H, W])
        coords = coords.to(device) # torch.Size([B, 1, 4096, 2])
        vis = vis.to(device) # torch.Size([B, 1, 4096])
        shift = shift.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        with torch.set_grad_enabled(phase == 'train'):
          # Get model outputs 
          outputs = model(inputs) # torch.Size([B, 3, H, W]) same as inputs shape
          outputs = outputs['out']
          outputs_coords, outputs_vis = get_predictions(device, outputs, coords, batch_size, shift=shift) # pre-processing of outputs

          # Zero out those in both outputs_coords and coords so they dont contribute to coords_loss
          outputs_coords_zeroed, coords_zeroed = zero_out_coords(coords, outputs_coords, vis)

          # Calculate loss
          sigmoid = nn.Sigmoid()
          vis_loss = (vis_criterion(sigmoid(outputs_vis), vis))/ 0.7
          coords_loss = (coords_criterion(outputs_coords_zeroed, coords_zeroed))/20
          loss = coords_loss + vis_loss

          # Backward + optimize in train phase
          if phase == 'train':
            loss.backward()
            # Gradient Value Clipping
            nn.utils.clip_grad_value_(model.parameters(), clip_value=clip)
            optimizer.step()

          # Statistics
          running_loss += loss.item() * inputs.size(0)
          running_vis_loss += vis_loss.item() * inputs.size(0)
          running_coords_loss += coords_loss.item() * inputs.size(0)

      epoch_loss = running_loss/total_samples
      epoch_vis_loss = running_vis_loss/total_samples
      epoch_coords_loss = running_coords_loss/total_samples

      # Log losses to train/val loss history
      if phase == 'train':
        train_loss_history.append((global_epoch+epoch, epoch_loss, epoch_coords_loss, epoch_vis_loss))
      if phase == 'val':
        val_loss_history.append((global_epoch+epoch, epoch_loss, epoch_coords_loss, epoch_vis_loss))
        # Deep copy the best validation model
        if epoch_loss < best_loss:
          best_loss = epoch_loss
          best_model_wts = copy.deepcopy(model.state_dict())

        # Save checkpoint
        save_checkpoint(model, best_model_wts, train_loss_history, val_loss_history, optimizer, batch_size, time.time()-since, description=description, path=path)

      
        
    # Report on losses of the model
    if verbose:
      print('-' * 50)
      print('Epoch {}/{}   {}'.format(global_epoch+epoch, global_epoch+(num_epochs - 1), datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))
      print('Train Loss: {:.4f} ({:.4f} coords loss, {:.4f} vis loss) '.format(train_loss_history[-1][1], train_loss_history[-1][2], train_loss_history[-1][3]))
      if val_loss_history:                
        print('Valid Loss: {:.4f} ({:.4f} coords loss, {:.4f} vis loss) '.format(val_loss_history[-1][1], val_loss_history[-1][2], val_loss_history[-1][3])) 
      time_elapsed = time.time() - since
      print('Time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    


  time_elapsed = time.time() - since
  print('-' * 50)
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Final Train Loss: {:4f}'.format(train_loss_history[-1][1]))
  print('Best Valid Loss: {:4f}'.format(best_loss))
  print('-' * 50)

  return model, best_model_wts, train_loss_history, val_loss_history, time_elapsed

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

def get_grid(device, N, B, H, W, shift=None, normalise=False):
  if shift==None:
    shift=torch.zeros(B, 2)
    shift = shift.to(device)

  N_ = np.sqrt(N).round().astype(np.int32)
  grid_y, grid_x = meshgrid2d(B, N_, N_, stack=False, device=device)
  grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16) + shift[:,0].reshape(B,1).tile(N).reshape(B,N)
  grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16) + shift[:,1].reshape(B,1).tile(N).reshape(B,N)

  if normalise:
    # normalise to values of range [-1, 1] - x = -1, y = -1 is the left-top pixel
    grid_x = (grid_x - W) / W 
    grid_y = (grid_y - H) / H

  xy = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2
  xy = xy.view(B, N_, N_, 2)

  return xy

def take_grid_points(device, coords, batch_size, outputs, shift):
  # Take the values of outputs from the NxN grid points
  grid = get_grid(device, coords.shape[2], batch_size, outputs.shape[2], outputs.shape[3], shift=shift, normalise=True) # ([B, H(64), W(64), 2])
  outputs = torch.nn.functional.grid_sample(outputs, grid, align_corners=False) # torch.size([B,2,H,W])

  # Reshape to match target shape
  outputs = torch.permute(outputs, (0, 2, 3, 1)) # torch.Size([B, H, W, 2])

  return outputs

def get_predictions(device, outputs, coords, batch_size, shift=None):
  # Take relevant points from the grid
  outputs = take_grid_points(device, coords, batch_size, outputs, shift)

  # Get coords predictions
  outputs_coords = outputs[:, :, :, :2]
  # Coordinates = predicted displacement + grid points in (B, 360, 640, 2)
  original_points = get_grid(device, coords.shape[2], batch_size, 360, 640)
  outputs_coords = original_points + outputs_coords
  outputs_coords = outputs_coords.view(batch_size,1,-1,2) # torch.Size([B, 1, 64*64, 2])

  # Get vis predictions
  outputs_vis = outputs[:, : , :, -1] # changing to tensor with 0s and 1s??
  outputs_vis = outputs_vis.view(batch_size,1,-1) # torch.Size([B, 1, 64*64])

  return outputs_coords, outputs_vis

def zero_out_coords(coords, outputs_coords, vis):
  # Take points that are not visible from vis ground truth data
  point_visiblity = vis > 0 # Boolean array with True if visible

  # Zero out those in both outputs_coords and coords so they dont contribute to coords_loss
  outputs_coords_zeroed = torch.zeros_like(outputs_coords)
  coords_zeroed = torch.zeros_like(coords)

  outputs_coords_zeroed[:, :, :, 0] = torch.where(point_visiblity, outputs_coords[:, :, :, 0], 0)
  outputs_coords_zeroed[:, :, :, 1] = torch.where(point_visiblity, outputs_coords[:, :, :, 1], 0)
  coords_zeroed[:, :, :, 0] = torch.where(point_visiblity, coords[:, :, :, 0], 0)
  coords_zeroed[:, :, :, 1] = torch.where(point_visiblity, coords[:, :, :, 1], 0)

  return coords_zeroed, outputs_coords_zeroed