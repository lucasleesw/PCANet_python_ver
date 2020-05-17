# PCANet_python_ver
SBU CSE512 Project

A simple implementation of PCANet: A Simple Deep Learning Baseline for Image Classification?
https://arxiv.org/abs/1404.3606

#### File list:

-   PCANet.py:

    Usage:

    -   Example: See interface.py and PCANet-MNIST.ipynb

    -   General use process:

    1. Load data into numpy ndarray(check data_loader.py as example)
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

-   interface.py

    A toy example of using PCANet

-   data_loader.py

    Simple data loader for MNIST and CIFAR10

    Sample usage:

    `train_images, train_labels, test_images, test_labels = load_mnist('path/to/mnist’)`

    `train_images, train_labels, test_images, test_labels = load_CIFAR10('path/to/CIFAR10’)`
    
-   Experiment Notebooks:

    -   PCANet_experiment_on_MNIST.ipynb

        A notebook shows the experiment results on MNIST.

    -   PCANet_experiment_on_CIFAR10.ipynb

        A notebook shows the experiment results on CIFAR10.

    -   VGG11_experiment_on_CIFAR10.ipynb

        For comparison, a notebook for training a simple VGG11 on CIFAR10.



#### Trained models

-  Google Drive link: [link to saved models](https://drive.google.com/open?id=1eiJ-Y1IsXHqPg32K8DfYvPIl7Dc5ZRmS)

The models are named by its training data sets, which means that each model is trained by the related data set. So, you will select the proper model depending on what testing data set you want to use.

-  Usage: Use `model = torch.load(PATH_to_the_model)` to load the model you need, then perform `prediction = model.predict(test_images)` to get the predicted results.

