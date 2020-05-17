# Entrance of program

from PCANet import *
from data_loader import *
import sys
from sklearn.metrics import accuracy_score
from sklearn import svm

train_images, train_labels, test_images, test_labels = load_mnist('data/MNIST')
test_train = (train_images[:10, :, :], train_labels[:10])

net = PCANet(k1=7, k2=7, L1=8, L2=8, block_size=7, overlapping_radio=0)

test_predict = test_images[:10, :, :]
prediction = net.predict(test_predict)
print(accuracy_score(test_labels[:10], prediction))
#
# cifar_train, cifar_train_labels, cifar_test, cifat_test_label = load_CIFAR10('data/cifar-10-batches-py')
# test_train = (cifar_train[:10, :, :, :], cifar_train_labels[:10])
# print(test_train[0].shape, test_train[1].shape)
#
# net = PCANet(k1=5, k2=5, L1=40, L2=8, block_size=8, overlapping_radio=0.5, spp_parm=(4, 2, 1), dim_reduction=1280)
#
# net.classifier = svm.LinearSVC(C=10)
#
# net.fit(*test_train)
# test_predict = cifar_test[:10, :, :]
# prediction = net.predict(test_predict)
# print('acc:', accuracy_score(cifat_test_label[:10], prediction))
