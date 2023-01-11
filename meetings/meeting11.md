## Notes ##
### Work done
* generating visibility data from PIPs and restructured dataset structure to have visibility data, and visualising this
* created training datasets (2 videos ~ 500 samples, 21 videos ~4800 samples) - generated coordinates of next frame seem to be correct
* train the prediction model (not on visibility data)

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
  |    |--- sample_id_0.npy
  |    |--- sample_id_8.npy
  |    |--- ... 
  |
  |---vis
       |--- sample_id_0.npy
       |--- sample_id_8.npy
       |--- ... 
```
__Visibility__

PIPs returns an array of floats for visiblity data (one for each tracked pixel). I made the assumption that negative numbers mean pixel is not visible,
which seems to be right (see visualisations of data), however few samples have all pixels with negative visibility (not sure what the reason is).

__Training__

When training, we can see train loss and validation loss at each epoch. Although losses decrease, the value is not satisfactory 
(both for bigger and smaller datasets the training loss stagnates around 24,000 - MLE loss). So the my prediction model fails to overfit.


### Questions
__1. What next?__

__2. Using a sequence?__

## Plan ##
* ?
