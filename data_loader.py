"""
Usage of data loader:
load_CIFAR10("path to CIFAR10")
load_mnist("path to mnist")
Returns train_images, train_labels, test_images, test_labels
in numpy array format.
"""
import os
import struct
import numpy as np
import pickle

def load_mnist(root):
    """Load MNIST data from `root`"""
    labels_path = os.path.join(root, "train-labels-idx1-ubyte")

    images_path = os.path.join(root, "train-images-idx3-ubyte")

    test_label_path = os.path.join(root, "t10k-labels-idx1-ubyte")

    test_images_path = os.path.join(root, "t10k-images-idx3-ubyte")

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28, 28)

    with open(test_label_path, 'rb') as tlbpath:
        magic, n = struct.unpack('>II', tlbpath.read(8))
        tlabels = np.fromfile(tlbpath, dtype=np.uint8)

    with open(test_images_path, 'rb') as timgpath:
        magic, num, rows, cols = struct.unpack('>IIII', timgpath.read(16))
        timages = np.fromfile(timgpath, dtype=np.uint8).reshape(len(tlabels), 28, 28)
    return images, labels, timages, tlabels

def load_mnist_2_0(dir_name):
    
      train_name = dir_name + "_train.amat"
      test_name = dir_name + "_test.amat"
    
      train_path = os.path.join(os.getcwd(),'data/mnist_2.0', dir_name, train_name)
      test_path = os.path.join(os.getcwd(),'data/mnist_2.0', dir_name, test_name)
    
      train_data = np.loadtxt(train_path)
      test_data = np.loadtxt(test_path)
    
      data_train = train_data[:, :-1] / 1.0
      data_train = data_train.reshape((data_train.shape[0],28,28))
      label_train = train_data[:, -1:].flatten()
    
      data_test = test_data[:, :-1] / 1.0
      data_test = data_test.reshape((data_test.shape[0],28,28))
      label_test = test_data[:, -1:].flatten()
      
      return data_train, label_train, data_test, label_test

def load_CIFAR_batch(filename):

    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """Load CIFAR10 data from `root`"""
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
