"""PCANet Class"""
import numpy as np
from scipy import signal
import gc

class PCANet:
    def __init__(self, k1, k2, L1, L2, block_size, overlapping_radio=0):
        # some parameter
        self.k1 = k1
        self.k2 = k2
        self.L1 = L1
        self.L2 = L2
        self.block_size = block_size
        self.overlapping_radio = overlapping_radio
        self.l1_filters = None
        self.l2_filters = None

    def mean_remove_img_patches(self, img, width, height):
        cap_x_i = np.empty((self.k1 * self.k2, width * height))
        idx = 0
        for i in range(width):
            for j in range(height):
                patten = img[i: i+self.k1, j:j+self.k2].flatten()
                cap_x_i[:, idx] = patten
                idx += 1
        cap_x_i -= np.mean(cap_x_i, axis=0)
        return cap_x_i

    def get_filter(self, train_data, num_filter):

        img_num, img_width, img_height = train_data.shape[0], train_data.shape[1], train_data.shape[2]
        patch_width = self.k1
        patch_height = self.k2
        img_patch_height = img_height - patch_height + 1
        img_patch_width = img_width-patch_width+1
        img_len = img_patch_height * img_patch_width
        cap_x = np.empty((patch_width * patch_height, img_len * img_num))
        # print(cap_x.shape)

        for n in range(img_num):
            cap_x[:, n*img_len:(n+1)*img_len] = self.mean_remove_img_patches(train_data[n], img_patch_width, img_patch_height)
            # print(cap_x_i.shape)
            if n % 100 == 0:
                gc.collect()

        vals, vecs = np.linalg.eig(np.matmul(cap_x, cap_x.T)/cap_x.shape[0])
        idx_w_l1 = np.argsort(vals)[:-(num_filter + 1):-1]
        cap_w_l1 = vecs[:, idx_w_l1]
        filters = cap_w_l1.T.reshape(num_filter, patch_width, patch_height)

        return filters

    def fit(self, train_data):
        self.l1_filters = self.get_filter(train_data, self.L1)
        print(self.l1_filters.shape)
        # print(train_data.shape)
        l1_conv_result = np.empty((train_data.shape[0]*self.l1_filters.shape[0], train_data.shape[1], train_data.shape[2]))
        l1_conv_idx = 0
        for image in train_data:
            for kernel in self.l1_filters:
                l1_conv_result[l1_conv_idx, :, :] = signal.convolve2d(image, kernel, 'same')
                l1_conv_idx += 1
        print(l1_conv_result.shape)
        self.l2_filters = self.get_filter(l1_conv_result, self.L2)
        print(self.l2_filters.shape)

