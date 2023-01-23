## Notes ##
### Work done
* training model on tiny datasets (20, 1) yielded loss of ~800, ~400 which is significantly less than before but still not good enough
* adding plotting of predcted and true coordinates while training

__Training on 1 sample__

Model is learning, but only ends up evenly spreading the points in the coordinates space. Paul suggested for the model to predict the displacement from the original point grid rather than the absolute coordinates -> This has reduced the loss. However, still needs a lot of training epochs to converge (>10,000).


### Questions
__1. How to reduce the loss?__

More training epochs, experiment with different losses.



## Plan ##
* experiment with different losses (Huber Loss)
* add prediction for visibility (use BCE Loss, mask loss calculation of coordinates for invisible points)
* add another frame by extending the channel dimension (need to duplicate weights, maybe allow to train weights in the first layer)
* over-train the modified models -> scale up
