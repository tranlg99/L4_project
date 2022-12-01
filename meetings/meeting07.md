## Notes ##
### Work done last week
* not much due to illness
* created pipeline for downloading data from youtube 8m and creating frames to ensure reproducibility
* put everything under VCS
* started pipeline for frames -> training data (cuda out or memory error when tracking 40x40 + points)

VCS has the structure of two folders one is a folder that user has to put into their drive, second is folder with python notebooks

### Questions


__1. How to solve the CUDA out of memory error?__

Try splitting the point to track grid, computing pixel trajectories on each sequence and then sticking them back together.



## Plan ##
* Fix bug of cuda out of memory
* Look into segmenation models
* After training data is generated, create a script that will visualise the data
