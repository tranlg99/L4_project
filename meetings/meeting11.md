## Notes ##
### Work done
* generating visibility data from PIPs and restructured dataset structure to have visibility data, and visualising this
* created training datasets (2 videos ~ 500 samples, 21 videos ~4800 samples) - generated coordinates of next frame seem to be correct
* train the prediction model (not on visibility data)

__Visibility__

PIPs returns an array of floats for visiblity data (one for each tracked pixel). I made the assumption that negative numbers mean pixel is not visible,
which seems to be right (see visualisations of data), however few samples have all pixels with negative visibility (not sure what the reason is).

__Training__

When training, we can see train loss and validation loss at each epoch. Although losses decrease, the value is not satisfactory 
(both for bigger and smaller datasets the training loss stagnates around 24,000 - MLE loss). So the my prediction model fails to overfit.


The model outputs `torch.Size([B, 2, 120, 240])` - I reshape the output to `torch.Size([B, 1, 120*240, 2])` and take the first 4096 points.

Coords: `torch.Size([B, 1, 4096, 2])`


### Questions
__1. Model training doesn't seem to overfit.__
Debug the code, reshaping was done wrongly + take output points from a grid

Try even smaller dataset

__2. Using a sequence?__
It is a good idea to use a sequence of frames rather than one which only gives stationary information. We can implement two frame input into the prediction model.

## Plan ##
* Debug and retrain on tiny dataset
* Input 2 frames instead of 1
