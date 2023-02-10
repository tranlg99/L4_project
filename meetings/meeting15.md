## Notes ##
### Work done
* double check copying of weights
* finetune last layer, + first layer, + second layer, all layers with SGD optimizer and lr=0.101 and lr=0.05
* investigation of spikes in the learning cureve: smaller lr, gradient value clipping
* investigation of poor performance: looking for bugs, feeding img in full resolution (matches coords resolution)

__Copying of weights__

I have double checked copying of weights by comparing the results of the old model (single input) and 
the new model with two inputs but with zeroed out weights in the second half of channels to not take into account the second input. 
With the same hyper-parameters I have trained with fine-tuning the last layer only 
and I’ve got training losses of 0.366441 vs 0.366437 respectively, and the resulting prediction looks the same. 
Therefore, I concluded that copying via of the weights `first_layer_weights.repeat(1, 2, 1, 1)` is correct.

__Performance of different models__

Next I have fine-tuned the two input model on
* last layer only
* first and last layer
* first, second and last layer
* and all layers

with SGD optimizer and learning rates of 0.101 and 0.05.
The best training loss is when we are training the whole network with lr=0.101, however it is not a big improvement from only training last layer.


__Debugging spikes in learning curves__

I had a question about spikes in the learning curves (especially when training first and second layer) and what could be the reason for them.

* I have tried lowering the learning rate but that only helped in the case with training all of the layers and it also worsened the performance.
* Paul said smaller learning rates and gradient clipping should help so I investigated the latter further.
* Additionally, make sure that batch-norm layers when overfitting single batches is either always in 'train' mode or always in 'eval' mode (i.e. don't switch the mode between training and validation phases, which is something one normally does, but can be dangerous with batchnorm with single-element batches)

__Debugging poor performance__

Paul has raised concerns about the quality of the predictions (very smooth and grid-like output). 
It should be overfitting the small random-looking shifts in the points much better than it is, given the amount epochs.
Therfore I have checked for bugs, e.g. the output resolution being too low or something...

The output has the same resolution as the input 120x240,
however the coords space have the resolution of the original image it was generated from 360x640.

Hence, I have created input samples of the same resolution as the coordinated space, which seems to make much better predicitons.

## Plan ##
* scale up - try around 10 frames (from one video and more videos)
* later tuning of hyper params (type of optimizer, num of epochs, lr)
* think about how to reconstruct frames for evaluation
* make sure to store models for later evaluation
* backgroung research and beginning of dissertation writing

