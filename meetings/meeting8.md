## Notes ##
### Work done last week
* pipeline for creating training data finished, however there are computation time issues
* research into semantic segmentation models
  * load a pretrained torchvision segmentation model (on colab), run it on some image (based on this [tutorial](https://debuggercafe.com/semantic-segmentation-using-pytorch-fcn-resnet/))



__Training data set generation pipeline__

Splitting and sticking back traj points together doesn't give the same results as computing on the whole point grid (from the PIPs paper, the computation of trajs is shared between particles within a video?). For generating 10x10 points and split in 10 chunks, the mean of difference abs(whole_grid-split_grid) is 0.049063973, max 0.3559265 and median 0.024650574. Is this error small enough? 

__Predition model inputs__

Images fed to FCN ResNet50 segmenatation model have to be normalized using `mean = [0.485, 0.456, 0.406]`
 and `std = [0.229, 0.224, 0.225]` - Whenever we use a pre-trained model for evaluation, then we have to use the mean and standard deviation on the dataset that it has been trained on. A single image has the format of `[batch_size x channels x height x width]`.
 
 
### Questions

__1. Generating the training dataset will take a lot of time (~430hrs). How to improve the performance?__
* reduce the resolution somewhat (e.g. 64x64 grid points)
* reduce the number of sets of frames per video (skip every other frame entirely (which might be good anyway to increase the amount of motion between frames), or skip every other chunk of eight frames)
* verify (if you haven't already) that it's the actual PIP computation that's slow, not reading the files or something else -- i.e. make sure the GPU is being used heavily
* 12 hours of running Colab when unattended?
* run it in chunks over ~four days with automated pipeline
* no need to have the entire dataset converted before continuing work on the prediction model

If it's really not practical to convert the whole dataset on colab, we can find some machine in SoCS to run this bit on.

__2. `Replacing the last layer of FCN ResNet50__

This is the architecture of the last layer of FCN ResNet 50, it gives output of shape `torch.Size([1, 21, 850, 1280])`
```
 (classifier): FCNHead(
    (0): Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Dropout(p=0.1, inplace=False)
    (4): Conv2d(512, 21, kernel_size=(1, 1), stride=(1, 1))
  )
```
Do we replace the whole classifier layer or just classifier[4]?

In this case, output should be torch.Size([1, N*N, 2]).
So we need to replace the last layer with something like torch.flatten(x, 1)? then nn.Linear(input_size, output_size)? and will need to tune the weights in this layer.

probably create a container like, something like in this [case](https://github.com/msminhas93/DeepLabv3FineTuning), where they replace FCNHead with DeepLabHead:

```
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )
        
model.classifier = DeepLabHead(2048, outputchannels)
```

## Plan ##
* investigate the reason for poor computational time (is it PIP computation or reading the files)
* try to convert the dataset in chunks
* visualisation of training data
* work on the prediction model - replacing the last layer, research on fine-tuning of resulting model
