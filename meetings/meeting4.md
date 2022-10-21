## Notes ##
### Work done last week
* understand the input and output of the PIPs model
* trying to feed it shorter videos (20 frames and with pixel that do not leave the shot) and few frames from DAVIS dataset
* started research in DL: MIT's introductory course on deep learning methods

After implementing PIPs model in Google Colab, I have tried shorter frames (20 frames) which the model will put into shorter sequences (8 frames in the demo code). The model is less confused and tracks the pixels well, especially the ones that keep being in the frame.
Still, the results of the model do not seem to be ideal, especially in driving datasets because of the texture (roads). For next week we need to choose a suitable dataset (not too different videos, preferably predictable movement in the videos, not many similar textures).


### Questions


1. __I tried to understand the input/output of PIP model. I am confused about the outputs of the model/not sure if these are even relevant__

  * Input: coordinates of points to track, the video (+ some optional parameters)
  * The model returns 4 (or 5) values: `coord_predictions, coord_predictions2, vis_e, losses`
  * `coord_preditictions` is a duplicate of `coord_predictions2` (in the demo they use the second one for animation of inference iterations)
  * Not sure what `vis_e` and `losses` is used for:
    * `vis_e` says if they pixel that is tracked is visible in the current frame
    * `losses` give the losses of the model, not relevant in our case


2. __What is the best structure to store the trajectories?__ 

  * The model outputs trajectories for each given tracking point. For example, if we feed it a 8 frame video with 256 tracking points, the model will output a tensor of shape `(1,8,256,2)`.
  * For our deep learning model we should store pairs of first rgb frames and coordinates of the pixels tracked in the eighth frame (if we are 8 frame sequences in the PIP model) in a np array file.

3. __What would be the ideal number of frames in a video to track a point on?__

  * Provided a pixel would go out of frame after a certain amount of frames, how long should be the videos for the training of the network
  * The lenght of the video can be determined later
  * As I wasn't able to change the sequence number of the PIP model, we will consider 8 frame videos for now


## Plan ##
* Choose a dataset as soon as possible! (Youtube dataset, Kinetics)
* Create a pipeline that parses data and stores the outputs which will be used to train the deep neural network (NumPy files of pairs of rgb frame and coordinates tensor)
* Continue research into Deep Learning: segmentation modesl, CNN classifiers
