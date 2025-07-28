# GAN_KMC

This is a project to learn 2D stochastic dynamics from KMC data using a GAN approach


## Code dependencies

The following code dependencies are required. Simple installation throgh pip should work just fine

* numpy
* pytorch
* torchvision
* matplotlib
* numba (optional)
* torchview


## Folder struture

The code is organized in the following folders:

* _data/_ contains the dataset exampples (better to be organized in subfolders
* _train_logs_ contains output of the training procedure
* _models/_ you can put trained models here
* _out/_ this folder is for the ouptut of the prediction/generated evolutions

## Scripts

The _train.py_ script can be used to train a model; use _python3 train.py --help_ to see all available options for the parser.
