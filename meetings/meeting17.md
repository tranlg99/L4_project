## Notes ##
### Work done
* reconstruction of frames using CT+NN interpolation
* trained model on 30 epochs on Colab -> not feasible in long term
* trained model on 200 epochs without data augmentation on uni gpu servers -> visualisation of results
* experimented with data augmentation
* started dissertation writing

### Frame reconstruction
I have implemented frame reconstruction with CT/NN interpolation with sampling from the grid or the whole output, results are [here](https://github.com/tranlg99/L4_project/blob/main/src/colab_notebooks/frame_reconstruction.ipynb).
Reconstructed frames used coordinate predictions of an overfitted model and also ground truth coordinates.

Using CT also produces smoother results in comparison with only NN but it also outputs RGB values outside of range 0-1 so I am clipping those.

Taking the network's predictions at every pixel produces smoother but less accurate reconstructions.I suppose it is because we are not training the weights associated to those pixels (we dont have ground truth PIPs data for them) therefore the model will keep their output displacement at 0 / they may have random noise.

To solve this, Paul suggested to vary the exact position of the PIPs grid such that it is not 'biased' - when choosing the points add a random offset of up to half the size of a grid cell to each.
Then store those sample locations in the dataset along with the displacements, so we know which positions to sample when calculating the loss. This might not be worth the effort so I suggested, running PIPs on higher resolution.

Another solution would be to feed randomly shifted (by up to half the grid spacing) versions of frames (and true displacements) during training. That way there si no need to run PIP at higher resolution, but still the network will not be seeing always the same set of grid points
Along with that more data augmenation would be to benefit for the model to generalise well.

### Model performance

Training on Colab is not feasible in long term as one epoch takes 20 minutes.

I have trained the model on uni stlinux12 server, one epoch takes 3.5 minutes - ran 200 epochs, results are in [this](https://github.com/tranlg99/L4_project/blob/main/src/colab_notebooks/model_1_results.ipynb) colab notebook.
Model does not seem to generalise well.

Dataset info: Train/valid/test 80/10/10% split, around 3500/450/450 samples.

### Questions
__1. Is my shifting logic correct?__

I am shifting both input frames and coords in dataloader (will make this random).

This means I have to start calculating the loss on the whole model output rather than just the grid or store the x,y shifting values to pass to the model so it knows where to calculate the loss.

__2. How to evaluate?__


## Plan ##
* finish data augmentation+shifting in dataloader
* train the model, will probably need more training epochs
* dissertation: introduction and background, (analysis)
