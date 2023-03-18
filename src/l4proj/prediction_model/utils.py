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
        grid_x = (grid_x - W/2) / (W/2)
        grid_y = (grid_y - H/2) / (H/2)

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
