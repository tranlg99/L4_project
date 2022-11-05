## Notes ##
### Work done last week
* research into CNNs and semantics segmentation: [MIT 6.S191: Convolutional Neural Networks](https://www.youtube.com/watch?v=uapdILWYTzE&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=3), [Overview of Semantic Image Segmentation](https://www.jeremyjordan.me/semantic-segmentation/)
* creating .jpg frames from videos in Colab and storing in Drive
* start a pipeline draft for frames -> training data


__CNN__ is a neural network used to learn and extract meaningful features, and learn hierarchy of features to learn and classify images. Fully connected networks unravels the input array, loses spatial information. Instead apply a series of convolutional filters to extract a feature volume, these are learned and apply activation function and downsample by pooling and repeat. Then output a set of feature volumes and feed them to a fully connected layer to perform classification task.

__Semantics segmentation__ aims to label each pixel of an image with a corresponding class of what is being represented (not separating instances of the same class; we only care about the category of each pixel). Can't use CNN architecture for classification because we need to preserve the spatial resolution, at the same time we can't preserve the full resolution throughout the whole network as it is computationally expensive. Need to downsample(convolutional layers) and upsample (transpose convolution).

### Generating frames from the videos
I created .jpg frames from the Taichi videos using code from [here](https://github.com/gsssrao/youtube-8m-videos-frames/blob/master/generateframesfromvideos.sh) in Colab (took around 20 mins for 132 videos). However, I had difficulties storing the frames directly in my Drive (it wouldnt appear on my Drive). Therefore, I created a .zip file and copied it onto my Drive (5.8GB).

### __Pipeline draft__
```
# mount drive to colab
# unzip file with frame folders - folder structure: frames/<video_name>/frames/*.jpg

# download model weights
# create pip model with these weights

# for each video feed its frames into pips model do:
	# create NumPy file
	# for each 8 frames of the video do:
		# run_model(model, rgbs, N, sw=None) -> pixel trajectories
		# add (rgb[0],trajs) to NumPy file - need to lower the resolution of the rgb
	# store NumPy file
	# cp NumPy file into drive -see if it works, maybe zip?
```



### Questions


__1. What type of CNN for semantics segmentation should I look into, U-Net?__

Paul reccomends torchvision, as it is well-written and implements many standard models. It includes several segmentation models, listed [here](https://pytorch.org/vision/stable/models.html#semantic-segmentation)


What to do:

1. create an instance of (e.g.) torchvision.models.segmentation.fcn_resnet50 and tell it to load pretrained weights

2. modify that model slightly, by removing its final layer and replacing with a 2D vector output at each pixel (instead of 20 class logits)

3. 'fine-tune' the model on your data -- i.e. start from the pretrained model, and modify its weights to better solve your task



torchvision itself has a discussion on how to modify and fine-tune their classification models [here](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html). For segmentation the principle is fairly similar; there is an example by a random github-person [here](https://github.com/msminhas93/DeepLabv3FineTuning), with an accompanying tutorial

For now (i) try to load a pretrained torchvision segmentation model (on colab), and run it on some image; (ii) try to understand what those tutorials on fine-tuning are talking about


__2. How many pixels should we track? ?x? grid__ 

  * maximum would be 360x640 pixels
  * 100x100 at least

```
    rgbs = rgbs.cuda().float() # B, S, C, H, W
    
    B, S, C, H, W = rgbs.shape
    rgbs_ = rgbs.reshape(B*S, C, H, W)
    H_, W_ = 360, 640
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    H, W = H_, W_
    rgbs = rgbs_.reshape(B, S, C, H, W)
```

Put everything under VCS, Gitlab will be the main source, everytime pull from git when working!

## Plan ##
* Create a pipeline that parses data and stores the outputs which will be used to train the deep neural network (NumPy files of pairs of rgb frame and coordinates tensor)
* Document the process of getting data (ensure reproducibility of the project), and generating training data
* __put my code under CVS!__
* After training data is generated, create a script that will visualise the data
