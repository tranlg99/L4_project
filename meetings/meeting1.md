## Notes ##
Paul revisited the project description, gave me datasets and relevant literature to look at. We discussed the project scope. 

Project will have 2 stages:

1. Reconstruct particle-based video approach to predict the motion of pixels in next few seconds
2. How to eliminate 'holes' introduced into the future frames

Further derivation from the project scope will be discussed. We can possibly look into comparison of different motion estimation approaches.

Driving video datasets:

* KITTI & KITTI-360
* Waymo Open Perception
* Audi A2D2
* nuScenes
* Lyft Level5

*Some of these require registration/approval to download; some of them are freely available. Paul used KITTI-360 and Waymo in the past; KITTI-360 is simpler and might be a good starting point.*

Weekly 30min meetings are preferred. For now Fridays 1300 at SAWb 502a.
Main communication via MSTeams.

## Plan ##
Mostly research and starting implementing particle-based video approach.

* Reading on keypoint tracking and optical flow methods [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629), [PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/abs/1709.02371)
* Look into possible datasets
* Use Colab to implement particle-based video approach from this paper [Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories](https://arxiv.org/abs/2204.04153)
* Reading on convolutional neural networks (CNNs) - [DL book](https://www.deeplearningbook.org/)
* Other reading: [Flow-Grounded Spatial-Temporal Video Prediction from Still Images](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yijun_Li_Flow-Grounded_Spatial-Temporal_Video_ECCV_2018_paper.pdf)
