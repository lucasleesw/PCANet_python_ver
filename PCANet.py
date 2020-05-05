"""PCANet Class"""
import numpy as np
from scipy import signal
import gc
import copy
from sklearn import svm
from sklearn.decomposition import PCA

# from guppy import hpy; h=hpy()


class PCANet:
    def __init__(self, k1, k2, L1, L2, block_size, overlapping_radio=0, linear_classifier='svm', spp_parm=None, dim_reduction=None):
        # some parameter
        self.k1 = k1
        self.k2 = k2
        self.L1 = L1
        self.L2 = L2
        self.block_size = block_size
        self.overlapping_radio = overlapping_radio
        self.l1_filters = None
        self.l2_filters = None
        if linear_classifier == 'svm':
            self.classifier = svm.SVC()
        else:
            self.classifier = None
        self.spp_parm = spp_parm
        if dim_reduction:
            self.dim_reduction = dim_reduction
        else:
            self.dim_reduction = None

    def mean_remove_img_patches(self, img, width, height):
        in_img = copy.deepcopy(img)
        del img
        cap_x_i = np.empty((self.k1 * self.k2, width * height))
        idx = 0
        for i in range(width):
            for j in range(height):
                patten = in_img[i: i + self.k1, j:j + self.k2].copy()
                cap_x_i[:, idx] = patten.flatten()
                idx += 1
        cap_x_i -= np.mean(cap_x_i, axis=0)
        return cap_x_i

    def get_filter(self, train_data, num_filter, rgb=False):
        if rgb: num_chn = train_data.shape[3]
        img_num, img_width, img_height = train_data.shape[0], train_data.shape[1], train_data.shape[2]
        patch_width = self.k1
        patch_height = self.k2
        img_patch_height = img_height - patch_height + 1
        img_patch_width = img_width - patch_width + 1
        if rgb:
            cap_c = np.zeros((num_chn * patch_width * patch_height, num_chn * patch_width * patch_height))
        else:
            cap_c = np.zeros((patch_width * patch_height, patch_width * patch_height))

        for n in range(img_num):
            if rgb:

                im = np.array([self.mean_remove_img_patches(train_data[n][:, :, i], img_patch_width, img_patch_height) for i in range(num_chn)]).reshape((num_chn * patch_width * patch_height, -1))

                cap_c += np.matmul(im, im.T)
            else:
                im = self.mean_remove_img_patches(train_data[n], img_patch_width, img_patch_height)
                cap_c += np.matmul(im, im.T)
            if n % 10000 == 0:
                print(n, 'th picture')
                gc.collect()
        # print(h.heap())
        vals, vecs = np.linalg.eig(cap_c / img_num * im.shape[1])

        idx_w_l1 = np.argsort(np.real(vals))[:-(num_filter + 1):-1]
        cap_w_l1 = np.real(vecs[:, idx_w_l1])
        # print(vecs)
        if rgb:
            filters = cap_w_l1.T.reshape(num_filter, patch_width, patch_height, num_chn)
        else:
            filters = cap_w_l1.T.reshape(num_filter, patch_width, patch_height)
        return filters

    def get_historgram(self, decimal_result):
        """ Useless! """
        histo_bins = range(2 ** self.L2)
        img_width, img_height = decimal_result.shape[1], decimal_result.shape[2]
        step_size = int(self.block_size * (1 - self.overlapping_radio))
        img_patch_height = img_height - self.block_size + 1
        img_patch_width = img_width - self.block_size + 1

        for l in range(self.L1):
            for i in range(0, img_patch_width, step_size):
                for j in range(0, img_patch_height, step_size):
                    patten = decimal_result[i: i + self.block_size, j:j + self.block_size]
                    histogram, _ = np.histogram(patten, histo_bins)

    def extract_features(self, img, rgb=False):
        if rgb:
            conv_result = np.empty((self.L1, self.L2, img.shape[0], img.shape[1]))

            for i in range(len(self.l1_filters)):
                l1_result = np.empty(img.shape)
                for ch in range(img.shape[2]):
                    l1_result[:, :, ch] = signal.convolve2d(img[:, :, ch], self.l1_filters[i, :, :, ch], 'same')
                l1_result = np.sum(l1_result, axis=-1)
                for j in range(len(self.l2_filters)):
                    conv_result[i, j, :, :] = signal.convolve2d(l1_result, self.l2_filters[j], 'same')
        else:
            conv_result = np.empty((self.L1, self.L2, img.shape[0], img.shape[1]))
            for i in range(len(self.l1_filters)):
                l1_result = signal.convolve2d(img, self.l1_filters[i], 'same')
                for j in range(len(self.l2_filters)):
                    conv_result[i, j, :, :] = signal.convolve2d(l1_result, self.l2_filters[j], 'same')
        # print(conv_result.shape)
        binary_result = np.where(conv_result > 0, 1, 0)
        # print(binary_result.shape)

        decimal_result = np.zeros((self.L1, img.shape[0], img.shape[1]))
        for i in range(len(self.l2_filters)):
            decimal_result += (2 ** i) * binary_result[:, i, :, :]

        histo_bins = range(2 ** self.L2 + 1)
        img_width, img_height = decimal_result.shape[1], decimal_result.shape[2]
        step_size = int(self.block_size * (1 - self.overlapping_radio))
        img_patch_height = img_height - self.block_size + 1
        img_patch_width = img_width - self.block_size + 1
        # print(decimal_result.shape)

        if self.spp_parm:
            feature_width = len(range(0, img_patch_width, step_size))
            feature_height = len(range(0, img_patch_height, step_size))
            feature = []
            for l in range(self.L1):
                before_spp = np.empty((feature_width, feature_height, len(histo_bins)-1))
                spp_idx_i = 0
                for i in range(0, img_patch_width, step_size):
                    spp_idx_j = 0
                    for j in range(0, img_patch_height, step_size):
                        patten = decimal_result[l, i: i + self.block_size, j:j + self.block_size]
                        histogram, _ = np.histogram(patten, histo_bins)
                        before_spp[spp_idx_i, spp_idx_j, :] = histogram
                        spp_idx_j += 1
                    spp_idx_i += 1
                after_spp = []
                for side in self.spp_parm:
                    W = feature_width // side
                    H = feature_height // side
                    for side_i in range(side):
                        for side_j in range(side):
                            after_spp.append(before_spp[side_i*W:(side_i+1)*W, side_j*H:(side_j+1)*H:, :].max(axis=(0, 1)))
                feature.append(after_spp)

            if self.dim_reduction:
                feature = np.array(feature).swapaxes(0, 1)
                dim_reduction_to = self.dim_reduction // feature.shape[1]
                after_pca = []
                for i in range(feature.shape[0]):
                    pca = PCA(n_components=dim_reduction_to, copy=False)
                    after_pca.append(pca.fit_transform(feature[i]))
                return np.array(after_pca).reshape((-1))
            else:
                return np.array(feature).reshape((-1))
        else:
            feature = []
            for l in range(self.L1):
                for i in range(0, img_patch_width, step_size):
                    for j in range(0, img_patch_height, step_size):
                        patten = decimal_result[l, i: i + self.block_size, j:j + self.block_size]
                        histogram, _ = np.histogram(patten, histo_bins)
                        feature.append(histogram)
            return np.array(feature).reshape((-1))

    def fit(self, train_data, train_labels):
        if len(train_data.shape) == 4:
            rgb = True
            num_chr = train_data.shape[3]
        else:
            rgb = False
        print('=' * 20)
        print('calculating L1_stage filters')
        self.l1_filters = self.get_filter(train_data, self.L1, rgb)
        print('shape of L1_stage filters:', self.l1_filters.shape)
        # print(train_data.shape)
        if rgb:
            l1_conv_result = np.empty(
                (train_data.shape[0] * self.l1_filters.shape[0], train_data.shape[1], train_data.shape[2], train_data.shape[3]))
        else:
            l1_conv_result = np.empty(
                (train_data.shape[0] * self.l1_filters.shape[0], train_data.shape[1], train_data.shape[2]))
        l1_conv_idx = 0
        # print(h.heap())
        for image in train_data:
            for kernel in self.l1_filters:
                if rgb:
                    for chn in range(num_chr):
                        l1_conv_result[l1_conv_idx, :, :, chn] = signal.convolve2d(image[:, :, chn], kernel[:, :, chn], 'same')
                else:
                    l1_conv_result[l1_conv_idx, :, :] = signal.convolve2d(image, kernel, 'same')
                l1_conv_idx += 1
        if rgb:
            l1_conv_result = np.sum(l1_conv_result, axis=-1)
        print('shape of L1 stage convolution results:', l1_conv_result.shape)

        print('=' * 20)
        print('calculating L2_stage filters')

        self.l2_filters = self.get_filter(l1_conv_result, self.L2)
        print('shape of L2_stage filters:', self.l2_filters.shape)

        print('=' * 20)
        features = []
        for i in range(len(train_data)):
            if i % 1000 == 0:
                print('extracting', i, 'th feature')
                gc.collect()
                # print(h.heap())
            feature = self.extract_features(train_data[i], rgb)
            features.append(feature)
        # print(h.heap())
        print('length of feature:', len(features[0]))
        print('='*20)
        print('features extracted, SVM training')
        self.classifier.fit(features, train_labels)
        # print(self.classifier.get_params())

    def predict(self, test_data):
        if len(test_data.shape) == 4:
            rgb = True
        else:
            rgb = False
        test_features = []
        print('=' * 20)
        for i in range(len(test_data)):
            if i % 500 == 0:
                print('predicting', i, 'th label')
            test_features.append(self.extract_features(test_data[i], rgb))
        predictions = self.classifier.predict(test_features)
        print('=' * 20)
        return predictions
