## Notes ##
### Work done
* added masking
* investigating SGD vs Adam optimizer when adding masking
* added second frame input to the model
* feeding one input to the model

__Masking__

1. take points that are visible from vis ground truth data: `point_visiblity = vis > 0 # Boolean array with True if visible`
2. zero out those in both outputs_coords and coords so they dont contribute to coords_loss
```
outputs_coords_zeroed = torch.zeros_like(outputs_coords)
coords_zeroed = torch.zeros_like(coords)
outputs_coords_zeroed[:, :, :, 0] = torch.where(point_visiblity, outputs_coords[:, :, :, 0], 0)
outputs_coords_zeroed[:, :, :, 1] = torch.where(point_visiblity, outputs_coords[:, :, :, 1], 0)
coords_zeroed[:, :, :, 0] = torch.where(point_visiblity, coords[:, :, :, 0], 0)
coords_zeroed[:, :, :, 1] = torch.where(point_visiblity, coords[:, :, :, 1], 0)
```

3. compute coord loss according to zeroed outpus: `coords_loss = (criterion(outputs_coords_zeroed, coords_zeroed))/20`


I compared performances of SGD and Adam optimizer - with normalised loss and  masking or no masking.

__Second frame__

1. change FCN_ResNet50 model's first layer to take 6 channel input: `nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)` and duplicate weights from the first layer 
```
  first_layer_weights = model_weights['backbone.conv1.weight'][0] # torch.Size([64, 3, 7, 7])
  doubled_weights = first_layer_weights.repeat(1, 2, 1, 1) # torch.Size([64, 6, 7, 7])
  
  # change the layer 
  modified_model.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

  # copy the doubled weights into first layer
  with torch.no_grad():
      modified_model.backbone.conv1.weight = torch.nn.Parameter(doubled_weights)
```
2. create dataset with 2 frame input (I take 0th and 3rd frame out of 8 frames) 
3. train the model

I compared performances of fine-tuning both first and last layer vs only last layer with both SGD/Adam optimizers.

### Questions


## Plan ##
* further overfitting and tuning of hyper params (type of optimizer, num of epochs, lr)
* scale up
