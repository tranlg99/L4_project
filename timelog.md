# Timelog

* 3D particle-based video prediction
* Linda Tran
* 2472123T
* Paul Henderson

## Guidance

* This file contains the time log for your project. It will be submitted along with your final dissertation.
* **YOU MUST KEEP THIS UP TO DATE AND UNDER VERSION CONTROL.**
* This timelog should be filled out honestly, regularly (daily) and accurately. It is for *your* benefit.
* Follow the structure provided, grouping time by weeks.  Quantise time to the half hour.

## Week 1
* Project bidding 

## Week 2
### 30 Sep 2022

* *1 hour* Read the abstract, introduction and conclusion of Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories paper
* *0.5 hour* Created GitLab repository for the project
* *0.5 hour* Initial meeting with supervisor
* *0.5 hour* Understanding learning outcomes and assesment criteria
* *0.5 hour* Making notes from initial meeting

### 2 Oct 2022
* *0.5 hour* Zotero reference manager set up
* *1 hour* Reading paper Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories (Section 3: PIPs), identifying further relevant research papers

## Week 3
### 5 Oct 2022
* *1 hour* Finishing reading Particle Video Revisited, familiarising with their [code base](https://github.com/aharley/pips)

### 6 Oct 2022
* *2 hours* Looking into Colab set up
* *0.5 hours* Adding project template from Moodle to repo

### 8 Oct 2022
* *0.5 hours* Meeting 2 with the supervisor
* *0.5 hours* Changing to GitHub and Linking Overleaf to GitHub

### 9 Oct 2022
* *1 hour* Notes from meeting 2, whole year plan update
* *4 hours* Implementing PIPs demo in Colab

## Week 4
### 12 Oct 2022
* *1 hour* Making notes from Particle Video Revisited paper
* *1 hour* Understanding their code base
* *1 hour* Looking into different datasets

### 13 Oct 2022
* *3 hours* Feeding a shorter driving video to the model

### 14 Oct 2022
* *0.5 hours* Meeting 3 with the supervisor

### 15 Oct 2022
* *0.5 hours* Notes from meeting 3

## Week 5
### 18 Oct 2022
* *1 hour* Research: [Intro to DL](https://www.youtube.com/watch?v=7sB052Pz0sQ&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)

### 20 Oct 2022
* *3 hours* Understanding output and inputs of PIPs model
* *1 hour* Downloading DAVIS and KITTI datasets

### 21 Oct 2022
* *1 hour* Trying to feed the model different videos (20 frame driving video, 80 frame breakdancing video)
* *1 hour* Meeting 4 with the supervisor, making notes

### 22 Oct 2022
* *1 hour* Looking into Youtube 8M and Kinetics dataset
* *1 hour* Converting short horse show from YT video into jpg frames and feeding it to the model

## Week 6
### 24 Oct 2022
* *1 hour* Converting short taichi video from YT into jpg frames and feeding it to the model
* *1 hour* Creating a series of outputs on different kinds of videos

### 25 Oct 2022
* *2 hours* Creating a series of outputs on different kinds of videos, different rate of frames per second, number of tracking points, sending results to Paul

### 27 Oct 2022
* *2 hours* Downloading 150 of Tai chi videos from Youtube 8M dataset using code from https://github.com/gsssrao/youtube-8m-videos-frames

### 28 Oct 2022
* *1 hour* Supervisor meeting and making notes

### 30 Oct 2022
* *3 hours* Research into [CNNs](https://www.youtube.com/watch?v=uapdILWYTzE&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=3) and [Semantic Segmenation models](https://www.jeremyjordan.me/semantic-segmentation/)

## Week 7
### 2 Nov 2022
* *0.5 hours* Creating a seperate google drive account for the data

### 3 Nov 2022
* *3 hours* Creating pipeline for video to frames conversion in Colab

### 4 Nov 2022
* *1 hour* Storing frames in Drive
* *1 hour* Supervisor meeting and making notes

## Week 8
### 7 Nov 2022
* *2 hours* Changing structure of the project to adhere to VCS. Change the main code storage from Colab Drive to Git Hub. When working on python notebooks, navigate to the notebook in Git Hub and then open the notebook in Colab using [this](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo) Google Chrome extension.
* *2 hours* Changing data download script. Now the full download dataset pipeline can be executed in Colab (downloading videos, generating frames).
* *2 hours* Creating pipeline for generating training data

### 8 Nov 2022
* *2 hours* Created pipeline and tried on few frames of a smaller dataset 

### 11 Nov 2022
* *1 hour* supervisor meeting + preparation for the meeting

## Week 9
### 15 Nov 2022
* *3 hours* debugging generate_training data script, still running into cuda out of memory error
* *0.5 hours* meeting 7 notes

### 16 Nov 2022
* *3 hours* solved cuda out of memery error, training data generation pipeline finished, however computation time will be an issue
* *2 hours* research into segmenation model, loading model in colab with pretrained weights, looking into replacing the final layer

### 17 Nov 2022
* *2 hours* running segmenation model on some image
* *0.5 hours* pre-meeting notes

### 18 Nov 2022
* *1 hour* supervisor meeting + making notes

## Week 10
### 24 Nov 2022
* *0.5 hours* checking access to stlinux12 uni server
* *3.5 hours* research into fine-tuning, modifying last layer of segmentation model to fit our task, creating training function

### 25 Nov 2022
* *1 hour* initial research into dataloaders, supervisor meet and making notes

## Week 10
### 30 Nov 2022
* *1 hour* Tutorial for writing custom datasets [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html), thinking about restructuring my dataset
### 1 Dec 2022
* *2 hours* Creating custom data set loader
* *2 hours* Restructuring training data set, revisiting generating of training dataset pipeline to fit
* *1 hour* Log for storing sample ids of already generated samples
* *3 hours* Testing data loading, generating small training data set
* *1 hour* Pre-meeting notes

## Week 12
### 12 Dec 2022
* *5 hours* Creating more diverse training data for justification of the chosed dataset

### 13 Dec 2022
* *2 hours* Status report draft
* *2 hours* Looking into other potential datasets

## Week 13-14
### 01 Dec 2023
* *5 hours* Adding visibility data to training dataset
* *2 hours* Changing structure of training data to accommodate visibility and not generating samples that have already been generated

### 02 Dec 2023
* *2 hours* Choosing appropriate videos for training dataset
* *4 hours* Creating dataset of 23 videos with visibility data

### 05 Jan 2023
* *2 hours* Showing visibility when plotting coords in dataloader.ipynb

### 09 Jan 2023
* *3 hours* Redefining prediction model for our data and splitting dataset into train and val

### 09 Jan 2023
* *2 hours* Reshaping coords and output to fit the model and computing loss correctly
* *1 hour* Running first training and evaluating

### 10 Jan 2023
* *1 hour* Creating a smaller dataset (2 videos) for the overfitting of the model

### 11 Jan 2023
* *0.5 hours* training the model on the smaller dataset
* *1.5 hour* notes, meeting
* *1 hour* debugging prediction model


## Week 15
### 16 Jan 2023
* *1 hr* Trying an even smaller dataset (20 samples)

### 18 Jan 2023
* *2 hrs* Investigation of the prediction model, research about Batch Normalization
* *0.5 hrs* Creating a dataset with one and two samples
* *2 hrs* Adding normalization of input images to the model and debugging

### 19 Jan 2023
* *2 hrs* Plotting predictions and true coordinates during training

### 20 Jan 2023
* *1 hr* Change model predictions from absolute coordinates to displacement vectors

## Week 16
### 23 Jan 2023
* *0.5 hrs* Meeting notes
* *2 hrs* Investigating different losses (MSE, Huber, L1)
* *2 hrs* Investigating different learning rates

### 26 Jan 2023
* *4 hrs* Adding prediction of visibility

### 25 Jan 2023
* *1 hrs* Debugging prediction of visibility (needed to add sigmoid layer due to BCE loss)
* *1 hrs* Meeting and meeting notes

### 29 Jan 2023
* *2 hrs* Adding masking of not visible points in the coords loss
* *2 hrs* Normalising vis loss and coords loss, investigating performance

## Week 16
### 30 Jan 2023
* *2 hrs* Adding masking of not visible points in the coords loss
* *2 hrs* Normalising vis loss and coords loss, investigating performance

### 01 Feb 2023
* *2 hrs* Trying Adam optimizer with masking and normalisation

### 02 Feb 2023
* *1 hr* creating a sample dataset with two input frames, changing my custom dataloader
* *1.5 hrs* creating a model that takes in two frames as input, copying weights

### 03 Feb 2023
* *2 hrs* trying adam vs sgd on two input model

## Week 17
### 06 Feb 2023
* 2 hrs* double checking copying of weights is correct

### 08 Feb 2023
* *4 hrs* fine-tuning on different layers of the network and evaluating

### 09 Feb 2023
* *2 hrs* debugging, investigating spikes with lowering learning rates, gradient clipping
* *2 hrs* feeding high resolution input to the model

### 10 Feb 2023
* *2 hrs* fine-tuning model with high res input on different layers
* *1 hrs* supervisor meet and notes

## Week 18
### 13 Feb 2023
* *4 hrs* training models on 10, 15 frames
* *1 hr* implementing checkpoints

### 14 Feb 2023
* *2 hrs* debugging, investigating spikes with lowering learning rates, gradient clipping
* *2 hrs* feeding high resolution input to the model

### 15 Feb 2023
* *5 hrs* background research
* *2 hr* full dataset generation

### 16 Feb 2023
* *3 hrs* reconstruction of frames using interpolation, trying on some coords
* *3 hrs* generating full dataset

### 17 Feb 2023
* *2 hrs* checking dataset
* *1 hr* supervisor meet, notes
* *1 hr* debug interpolation, now using NN interpolation

### 18 Feb 2023
* *3 hrs* regenerating corrupted frames
* *1 hr* exclude invisible points from reconstrucion of frames

### 19 Feb 2023
* *3 hrs* dissertation writing: introduction
* *1 hr* training model on Colab
* *2 hrs* reconstruction of frames using CT interpolation

## Week 17
### 20 Feb 2023
* *3 hrs* training model on Colab

### 21 Feb 2023
* *2 hrs* dissertation writing: background

### 22 Feb 2023
* *5 hrs* setting up uni stlinux12 server, creating environment, downloading dataset, writing code for running the model
* *2 hrs* training model

### 23 Feb 2023
* *2 hrs* training model 1 (without data augmentation) 200 epochs and visualisation of results
* *2 hrs* data augmenation
* *1 hr* dissertation writing: background

### 24 Feb 2023
* *2 hrs* data augmentation: shifting
* *1 hr* supervisor meet

## Week 18
### 27 Feb 2023
* *2 hrs* traning models with data augmenetation and shifting

### 28 Feb 2023
* *2 hr* dissertation writing: introduction
* *3 hrs* dissertation writing: background

### 1 Feb 2023
* *4 hrs* dissertation writing: background

### 2 Feb 2023
* *2 hrs* investigeting batchnorm and drop out
* *2 hrs* loading model2, model3, model4 weights into colab and visualising 


