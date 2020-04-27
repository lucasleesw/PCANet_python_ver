# Entence of program

import PCANet
from data_loader import *
import sys

# print(load_CIFAR10('data/cifar-10-batches-py'))

"""
Usage of data loader:
load_CIFAR10("path to CIFAR10")
load_mnist("path to mnist")
Returns train_images, train_labels, test_images, test_labels
in numpy array format.
"""
train_images, train_labels, test_images, test_labels = load_mnist('data/MNIST')