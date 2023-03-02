## Notes ##
### Work done
* training models with data augmentations and shifting (200 epochs)
* batchnorm layer doesnt seem to be active
* implemented drop out layers in last and second to last layers, now in training (p=0.2)
* finished introduction and background of dissertation

### Models with data augmentations and shifting

I show quantitative results on 4 models (without any data augmentation or shifting, with shifting, with data augmentation, with shifting and data augmentation).

Models do not show satisfying predictions just yet. But models with shifting make finer-grained frame reconstructions.

### Drop out layer

Now training two models with drop-out layer only in the last layer and last and second to last layer. Training will be done in the evening of 03/03.

### Questions
__1. Batch norm is not changing in both eval and train mode, why?__


__2. What to change for better performance?__
* Drop-out layers
* Longer training period? Might only overfit the model
* Getting more data? Changing dataset?

## Plan ##
* finish training of models with drop-out layers
* evaluation script
* dissertation: analysis, design
