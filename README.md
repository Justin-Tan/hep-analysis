# hep-analysis
hep-analysis aims to reproduce the traditional data-analysis workflow in high-energy physics using Python. This allows us to employ the wide range of modern, flexible packages for machine learning and statistical analysis written in Python to increase the sensitivity of our analysis.

![Mbc distribution](plots/readme_plots/mbc.png?raw=true "Sample feature distribution")
![NN output](plots/readme_plots/nn_prob.png?raw=true "Neural Network Output")

## Functionality
### Machine Learning
#### Models
- [x] Deep Neural Networks - `TensorFlow`, `PyTorch`
- [x] Recurrent Neural Networks - `TensorFlow`
- [x] Gradient Boosted Trees - `XGBoost`, `LightGBM`
- [x] Hyperparameter Optimization - `scikit-learn`, `HyperBand`

#### Training/Ensembling
- [x] Multi-GPU
- [ ] Distributed Training
- [ ] Smart Model Ensembles

### Statistical Analysis
- [ ] Signal Yield Extraction
  - [ ] Maximum Likelihood
  - [ ] Bayesian Approach
- [ ] Precision Branching Fraction / CP Asymmetry Measurements

## Notebooks
To view the notebooks available in this repo, visit http://nbviewer.jupyter.org/github/Justin-Tan/hep-analysis/tree/master/notebooks and navigate to the relevant section. 

## Getting Started
Under development.
```
$ git clone https://github.com/Justin-Tan/hep-analysis.git
$ cd hep-analysis/path/to/project
# Check command line arguments
$ python3 main.py -h
# Run
$ python3 main.py <args> 
```

## Dependencies
The easiest way to install the majority of the required packages is through an [Anaconda environment](https://www.continuum.io/downloads). Download (if working remotely use `wget`) and run the installer. We recommend creating a separate environment before installing the necessary dependencies.

#### Python 3.6
```
$ conda create -n hep-ml python=3.6 anaconda
```
#### ROOT
Install the binaries from [here](https://root.cern.ch/downloading-root).

#### root_numpy
[Installation instructions](http://scikit-hep.org/root_numpy/install.html). To install in your home directory:
```
$ pip install --user root_numpy
```
#### TensorFlow 1.2
[Installation instructions](https://www.tensorflow.org/install/install_sources). We recommend building from source for better performance. If GPU acceleration is required, install the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn). Note: As of TF 1.2.0, your code must be modified for multi-gpu support. See the multi-gpu implementations for sample usage (tested on a Slurm cluster, but should be fully general).

#### PyTorch
[Installation instructions](https://github.com/pytorch/pytorch#installation).
```
# using Anaconda
$ conda install pytorch torchvision -c soumith
```

#### XGBoost 
Clone the repo as detailed [here](http://xgboost.readthedocs.io/en/latest/build.html)

## More Information
* [TensorFlow](https://www.tensorflow.org/). Open-source deep learning framework.
* [PyTorch](https://github.com/pytorch/pytorch) Dynamic graph construction!
* [XGBoost](http://xgboost.readthedocs.io/en/latest/). Extreme gradient boosting classifier. Fast.
* [root_numpy](https://github.com/scikit-hep/root_numpy). Bridge between ROOT and Python.
* [Hyperband](https://people.eecs.berkeley.edu/~kjamieson/hyperband.html?utm_content=buffera95c2&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer).
Bandit-based approach to hyperparameter optimization.
