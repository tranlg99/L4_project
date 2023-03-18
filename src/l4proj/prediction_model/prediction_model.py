import copy
from datetime import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

from .checkpoints import save_checkpoint
from .utils import get_predictions, zero_out_coords


def initialise_model(device, to_learn=[]):
  """
  Initialize torchvision.models.segmentation.fcn_resnet50 model with the best available weights
  and modify for our purposes, specify what parameters to fine-tune in to_learn arg

  """
  weights = FCN_ResNet50_Weights.DEFAULT
  modified_model = fcn_resnet50(weights=weights)

  # Get first layer weights and double
  model_weights = {k: (v, v.dtype, v.shape) for k, v in modified_model.state_dict().items()}
  first_layer_weights = model_weights['backbone.conv1.weight'][0]
  doubled_weights = first_layer_weights.repeat(1, 2, 1, 1)


  # Modify the model by doubling input channel from 3 to 6 and copy the doubled weights into first layer
  modified_model.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  with torch.no_grad():
      modified_model.backbone.conv1.weight = torch.nn.Parameter(doubled_weights)

  # Removing final layer, instead of torch.Size([1, 21, 360, 640]) -> torch.Size([1, 3, 360, 640])
  modified_model.classifier[3] = nn.Sequential()
  modified_model.classifier[4] = nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1))

  # Choose which params to tune
  if "all" not in to_learn:
    for name, param in modified_model.named_parameters():
      if name not in to_learn:
        param.requires_grad = False

  # Load model onto computation devicce
  modified_model.to(device)
  print("Model initialised")

  return modified_model

def train_model(device, model, dataloaders, optimizer, past_t_loss, past_v_loss, best_weights =[], num_epochs = 100, verbose = True, clip = 1.0, eval_freq=5, description="", path=""):
  """
  Train function
  Args:
    device (string): computation device
    model (nn.Module): model to train
    dataloaders (dictionary): dictionary with 'train' and 'val' dataloaders
    past_t_loss, past_v_loss (list): lists of past train and validation loss to store in checkpoint
    best_weights (list, optional): best weights of previous model, if resuming training
    num_epochs (integer, optional): number of epochs to train
    verbose (Bool, optional): print train log statements
    clip (float, optional): gradient clip value,
    eval_freq (integer, optional): frequency of running validation during training,
    description (string): model description to store checkpoint as
    path (string, optional): path to folder where to store checkpoints
  """
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
          vis_loss = (vis_criterion(sigmoid(outputs_vis), vis))*20
          coords_loss = (coords_criterion(outputs_coords_zeroed, coords_zeroed))/25
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
