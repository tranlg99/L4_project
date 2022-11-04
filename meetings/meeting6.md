## Notes ##
### Work done last week
* research into CNNs and semantics segmentation: [source1](https://www.youtube.com/watch?v=uapdILWYTzE&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=3), [source2](https://www.jeremyjordan.me/semantic-segmentation/)
* creating .jpg frames from videos in Colab and storing in Drive
<!---* start a pipeline draft for frames -> training data--->


### Questions


__1. What type of CNN for semantics segmentation should I look into, U-Net?__

  * ???


__2. How many pixels should we track? ?x? grid__ 

  * ???

```
    rgbs = rgbs.cuda().float() # B, S, C, H, W
    
    B, S, C, H, W = rgbs.shape
    rgbs_ = rgbs.reshape(B*S, C, H, W)
    H_, W_ = 360, 640
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    H, W = H_, W_
    rgbs = rgbs_.reshape(B, S, C, H, W)
```

<!---
__3. Question?__

  * ???
--->

## Plan ##
* Create a pipeline that parses data and stores the outputs which will be used to train the deep neural network (NumPy files of pairs of rgb frame and coordinates tensor)
* Document the process of getting data (ensure reproducibility of the project), and generating training data
