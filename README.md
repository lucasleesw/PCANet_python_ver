# PCANet_python_ver
SBU CSE512 Project

A simple implementation of PCANet: A Simple Deep Learning Baseline for Image Classification?
https://arxiv.org/abs/1404.3606

Usage of PCANet:

- Example: See interface.py and PCANet-MNIST.ipynb

- General use process:

  1. Load data into numpy ndarray(check data_loader2.py as example)
  2. Initialize PCANet class, e.g.:

  `net = PCANet(k1=7, k2=7, L1=8, L2=8, block_size=7, *other_parameters)`

  Parameters:
  ```python
  k1
  # the mean remove patch size width, required
  k2
  # the mean remove patch size height, required
  L1
  # the number of filters in the First stage, required
  L2
  # the number of filters in the Second stage, required
  block_size
  # the block size of histogram, required
  overlapping_radio=0 
  # overlapping radio, 1-0, optional, default 0
  linear_classifier='svm' 
  # linear_classifier, a linear_classifer or 'svm', 'svm' means sklearn.svm.SVC(), optional, default 'svm'
  spp_parm=None
  # parameters for spp, needs to be a list, eg[4,2,1], optional, default none
  dim_reduction=None
  # dim reduction after spp, the number of dimension reduce to, optional, default none
  ```
  
  3.  Train PCANet with train data, e.g.:
  
  `net.fit(train_data)`
  
  4.  Predict test data's label with trained PCANet, e.g.:
  
  `prediction = net.predict(test_data)`




Pre-trained modes:

# todo

