# PLSGPU
This repository provides a GPU implementation of the Partial Least Squares (PLS) algorithm. 


## Requirements

- [Scikit-learn](http://scikit-learn.org/stable/)
- [Keras](https://github.com/fchollet/keras)
- [Tensorflow](https://www.tensorflow.org/) 
- [Python 3](https://www.python.org/)

## Quick Start
[main.py](main.py) provides an example of usage of the PLS GPU. Currently, we do not implement the learning stage, therefore, you need to learn a PLS model using scikit-learn (which is performed in CPU).
Finally, by experiments, we note that for a small number of samples (for example., 100) the CPU version is slightly faster.
According to figure, our GPU implementation of the PLS achieves considerable speed-up regarding the CPU version.
![](Figures/plot.png)

## Parameters
Our PLSGPU method takes two parameters:
1. A PLS model learn from scikit-learn (as aforementioned)
2. Batch size. This parameter controls the number of samples sent to GPU. Note that, larger batch sizes faster the prediction.
