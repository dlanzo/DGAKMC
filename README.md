# Deep Generative Adversarial Kinetic Monte Carlo

This project aims to learn stochastic dynamics from KMC data using a GAN approach. Some methods/functions may also be found in the [CRANE](https://github.com/dlanzo/CRANE) project. However, the latter is not required for this project to run.


## Code dependencies

The following code dependencies are required. Simple installation through pip (possibly with virtualenvs) should work just fine

* numpy
* pytorch
* torchvision
* matplotlib
* numba (optional)
* torchview


## Folder structure

The code is organized in the following folders:

* _data/_ contains the dataset examples (better to be organized in subfolders)
* _train_logs_ contains output of the training procedure
* _models/_ you can put trained models here
* _out/_ this folder is for the ouptut of the prediction/generated evolutions
* _src/_ this foldet contains the source code, distributed into the _classes_, _convolutions_, _dataloaders_, _parser_ and _utils_ modules

## Scripts

The _train.py_ script can be used to train a model; use _python3 train.py --help_ to see all available options for the parser. Some features are implemented (e.g. LSGAN, WGAN) but have not been fully experimented yet. If you are interested in using those, or would like to collaborate in possible implementations, please don't hesitate to contact us.

## Datasets

A dataset of KMC simulations of monoatomic step dyanmics on a simple cubic 100 surface is available at [MaterialsCloud](https://doi.org/10.24435/materialscloud:8j-b8). In the same repository, some GAN-generated trajectories are also available.

## Publications

The present code has been used in the paper "Learning Kinetic Monte Carlo stochastic dynamics with Deep Generative Adversarial Networks" ([preprint](https://doi.org/10.48550/arXiv.2507.21763)). The dataset and generated trajectories are available [here](https://doi.org/10.24435/materialscloud:8j-b8).
