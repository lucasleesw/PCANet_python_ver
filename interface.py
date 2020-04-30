# Entrance of program

from PCANet import *
from data_loader import *
import sys

# print(load_CIFAR10('data/cifar-10-batches-py'))


train_images, train_labels, test_images, test_labels = load_mnist('data/MNIST')
test_train = (train_images[:1000, :, :], train_labels[:1000])
print(test_train[0].shape, test_train[1].shape)
net = PCANet(k1=7, k2=7, L1=8, L2=8, block_size=7, overlapping_radio=0)
net.fit(*test_train)
test_predict = test_images[:5, :, :]
prediction = net.predict(test_predict)
print(prediction, test_labels[:5])
