## Notes ##
### Work done last week
* pipeline for creating training data finished
* research into semantic segmentation models - fine-tuning
  * load a pretrained torchvision segmentation model (on colab), run it on some image

Splitting and sticking back traj points together doesn't give the same results as computing on the whole point grid (in PIPs, the computation of trajs is shared between particles within a video). For generating 100 points and split in 10 chunks, the mean of difference (whole_grid-split_grid) is 0.0013367033, max 0.30690002 and median 0.00030517578. Is this error small enough? 


### Questions

__1. Generating the training dataset will take a lot of time (~430hrs). How to improve the performance?__
* reduce the resolution somewhat (e.g. 64x64 grid points)
* reduce the number of sets of frames per video (skip every other frame entirely (which might be good anyway to increase the amount of motion between frames), or skip every other chunk of eight frames)

* verify (if you haven't already) that it's the actual PIP computation that's slow, not reading the files or something else -- i.e. make sure the GPU is being used heavily

* 12 hours of running Colab when unattended?
* run it in chunks over ~four days with automated pipeline
* no need to have the entire dataset converted before continuing work on the prediction model

If it's really not practical to convert the whole dataset on colab, we can find some machine in SoCS to run this bit on.

## Plan ##
* try to convert the dataset in chunks
* work on prediction model
* visualisation of training data
