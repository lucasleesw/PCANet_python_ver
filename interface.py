# Entrance of program

from PCANet import *
from data_loader import *
import sys

# print(load_CIFAR10('data/cifar-10-batches-py'))


train_images, train_labels, test_images, test_labels = load_mnist('data/MNIST')
test_train = train_images[:10, :, :]
print(test_train.shape)
net = PCANet(k1=7, k2=7, L1=8, L2=8, block_size=7)
net.fit(test_train)