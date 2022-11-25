## Notes ##
### Work done last week
* check access to stlinux12
* investigated running times of PIP
* trying segmentation model on one of our image
* modifying segmentation model
* research and implementation of [fine-tuning](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html) (still need to define dataloader, auxilary output)

__Running times__

It takes 13-15s for to calculate the 8th frame position of 100x100 points divided into 7x~1500points chunks.
It takes 2-3s to calculate the same thing for 1600 points without dividing the grid.
-> poor performance does not come from reading/writing files, GPU is utilised

Next step is to decrease the frequency of sampling from the video frames, skip every second one, etc. and create a small training data set (10 videos).



__Access to uni servers with GPUs__

I have access to stlinux12.
There might be an issue with lack of space, I have contacted IT services to allocate me more space (classmate was allocated 5G).
Servers have Python 3.6.8, might be annoying with some packages.



__Modified model & fine-tuning__

In feature extraction, only the weights of final layer of the pretrained model are updated.
In finetuning, we update all of the model’s parameters for our new task.

Last layer has been modified to output `torch.Size([1, 2, 120, 240])`

```
(classifier): Sequential(
    (0): Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
  )
  (aux_classifier): Sequential(
    (0): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(216, 2, kernel_size=(1, 1), stride=(1, 1))
  )
  
my_optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
my_criterion = nn.CrossEntropyLoss()
```

### Questions

__2. Do I use auxilary output for training?__

__1. I need to create my custom dataset in a way so I can use Dataloader, to make it easier to work with the data. Any tips on this? __


## Plan ##
* define dataloader
* get a small training dataset (10 videos)
* visualisation of training data in a different notebook

