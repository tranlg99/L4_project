# L4 Project: Source code

Code consists of two folders: `l4proj/` and `colab_notebooks/`.

`l4proj/` is source code used for training our models on University of Glasgow Linux servers.

`colab_notebooks/` folder includes all python notebooks (training data generation, evaluation of models) and a `drive_folder` with all executables that should be copied into your Google Drive in order to successfully run the notebooks.


## Run instructions


### Datasets
#### Synthetic dataset download links
* Raw video frames: [Frames](https://drive.google.com/file/d/1sd8PyA7L4yCVBOyiOs7FeFUGNwXvOnnf/view?usp=share_link)
* Train datasets: [Part 1](https://drive.google.com/file/d/1-78gwyPbF8r03F6EyrBrlK1z6qWJwgE2/view?usp=share_link), [Part 2](https://drive.google.com/file/d/1-34uUmisCzUC616s3Ctv4Ogt7hEBr7I-/view?usp=share_link), [Part 3](https://drive.google.com/file/d/1lQD84SfvnVBgbq7jiB8FNAN07KtmCGcA/view?usp=share_link) 
* Validation dataset: [Valid](https://drive.google.com/file/d/1U7fpiq253KjfPiI8GXtyVq7_oFmvgOMN/view?usp=share_link)
* Test dataset: [Test](https://drive.google.com/file/d/1-AtP2n5N0J7XTRzPlOz95eomtsLbuyvR/view?usp=share_link)

#### Natural dataset download links
* Raw video frames: [Frames](https://drive.google.com/file/d/1YxcCTaXGCVr9Iv-YoRJFmb-2_5qUFoPo/view?usp=sharing)
* Train datasets: [Train](https://drive.google.com/file/d/1oURaoO0YJm30cRPqUPaSUmHMcqcGSm81/view?usp=share_link)
* Validation dataset: [Valid](https://drive.google.com/file/d/1-0aDjjgh0RCRlWFYmNxIApk-wJ6p8mq_/view?usp=share_link)
* Test dataset: [Test](https://drive.google.com/file/d/1-AtP2n5N0J7XTRzPlOz95eomtsLbuyvR/view?usp=share_link)


### Run models

Copy `drive_folder/` and put it in your Google Drive account.
Run `.ipynb` files

### Train models


### Requirements

* Python 3.8.10
* Packages: listed in `l4proj/requirements.txt` 
* Models trained on University of Glasgow Linux Server stlinux12 with 2 GeForce GTX 1080 GPUs
