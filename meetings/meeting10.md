## Notes ##
### Work done last week
* created custom dataset structure and loader (based on [this tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html))
* restructured training data and its generation, log file to track sample ids
* tested dataloading of my custom dataset (batches)
* crated a dataset from one video (381 samples)

__Custom Dataset Structure__
```
training_data
  |
  |--- sample_ids.txt
  |
  |--- frame0
  |    |--- sample_id_0.npy
  |    |--- sample_id_8.npy
  |    |--- ...     
  |
  |---frame1
  |    |--- sample_id_0.npy
  |    |--- sample_id_8.npy
  |    |--- ... 
  |
  |---coords
       |--- sample_id_0.npy
       |--- sample_id_8.npy
       |--- ... 
```
`sample_ids.txt` serves as a list of sample_ids as well as a log to keep track of what training data I have generated already.

`frame0/` directory stores the first frames of the sequence (training data input)

`frame1/` directory stores the last frames of the sequence (sanity check for generated training data)

`coords/` directory stores the coordinates of pixels in the last frame



### Questions
__1. Is this structure appropriate? Especially, in terms of storage.__
If zipped, it should be okay.


__2. For the visualisation of the data, tracking points and image frame have different resolutions in the training data.__
Not a big problem. Better, store smaller resolution and resize image later, because of storage issues.


## Plan ##
* revisit number of tracking points (change to 64x64), dataset structure (will need to add visibility data)
* make other samples, to see the credibility of person moving in frames
* if all is good then get a small training dataset (of 10 videos)
* visualisation of training data in a different notebook
* fine-tune modified prediction model on this data
