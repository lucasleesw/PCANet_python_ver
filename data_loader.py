import os
import struct
import numpy as np
import pickle

def load_mnist(root):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(root, "train-labels-idx1-ubyte")

    images_path = os.path.join(root, "train-images-idx3-ubyte")

    test_label_path = os.path.join(root,"t10k-labels-idx1-ubyte")

    test_images_path = os.path.join(root,"t10k-images-idx3-ubyte")

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    # 读入magic是一个文件协议的描述,也是调用fromfile 方法将字节读入NumPy的array之前在文件缓冲中的item数(n).

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28, 28)

    with open(test_label_path, 'rb') as tlbpath:
        magic, n = struct.unpack('>II', tlbpath.read(8))
        tlabels = np.fromfile(tlbpath, dtype=np.uint8)
    # 读入magic是一个文件协议的描述,也是调用fromfile 方法将字节读入NumPy的array之前在文件缓冲中的item数(n).

    with open(test_images_path, 'rb') as timgpath:
        magic, num, rows, cols = struct.unpack('>IIII', timgpath.read(16))
        timages = np.fromfile(timgpath, dtype=np.uint8).reshape(len(tlabels), 28, 28)
    return images, labels, timages, tlabels



def load_CIFAR_batch(filename):
    """ 载入cifar数据集的一个batch """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ 载入cifar全部数据 """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs) # 使变成行向量,最终Xtr的尺寸为(50000,32,32,3)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte