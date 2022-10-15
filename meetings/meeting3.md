## Notes ##
I updated Paul on my work from last week: implementing a particle video model, feeding it some data.

I have succesfully implemented their model in Google Colab, and tried it on their demo data (video of a dog - 101 frames).
Then I looked into some driving datasets, got access to KITTI data.
Before getting access to KITI, I downloaded [this driving dataset](https://drivingstereo-dataset.github.io/) to test in on the model.
However, when feeding it longer videos (142 frames) pixels frequently went out of frame which made the model confused.


I had questions about:

1. __Model not working optimally on pixels that go out of frame;__ 

  * Paul asked to investigate and compare how the model parses my data at each stage against their demo data.
  * We can possibly look into other datasets (animals, etc.) where the tracking point doesn't go out of the frame so often.
  * Or use shorter videos.


2. __Provided the model will work optimally and dataset is chosen, what would be next?__

    Create a pipeline that automatically parses data that I have uploaded to the drive and stores the outputs which will be used to train the deep neural network.




## Plan ##
* Try shorter videos, tracking different points, investigate resolution of the frames, etc.
* Possibly look into other datasets
* Understand input, output of their model
* Begin to research Deep Learning: segmentation modesl, CNN classifiers
