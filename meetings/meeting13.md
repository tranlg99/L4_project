## Notes ##
### Work done
* investigating with different losses and learning rates
* add prediction for visibility (use BCE Loss)

__Loss investigation__

I investigeted Huber Loss (with lr=0.1, 0.01, 0.2, 0.05, 0.4, 0.25), MSE loss (with lr=0.1, 0.01, 0.2, 0.15, 0.075, ..., 0.101), L1 loss with 10_000 epochs.

With Huber loss, learning converges slower so I used higher learning rates. However, predictions are not better than with MSE. So I decided to use MSE with lr=0.101.


### Questions
__1. Model does not learn with visibility loss?__

Last convulation layer of the model now outputs torch.Size([B, 3, H, W]). I sample the points that are on the grid (64x64), 
take the first two channels as xy displacent vectors of the predicted coordinates. Take the last channel tensor as visibility predictions 
-> create a tensor with only 0,1s `torch.where(outputs[:, : , :, -1] > 0, 1.0, 0.0`

then add up the losses for the optimizer

`loss = criterion(outputs_coords, coords) + vis_criterion(outputs_vis, vis) # TO DO: normalise losses`

Sources say to add nn.Sigmoid before applying BCE Loss which helped.

__2. How to mask coords loss for not visible points?__

From ground truth data see which points are not visible and zero out that coords loss.

## Plan ##
* normalise losses, mask loss calculation of coordinates for invisible points?
* add another frame by extending the channel dimension (need to duplicate weights, maybe allow to train weights in the first layer)
* over-train the modified models -> scale up
