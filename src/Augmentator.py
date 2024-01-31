import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.Preparer import Preparer


class Augmentator:
    def __init__(self, dir_path='./label_data', d_a=10, half_size=128, kernel_size=15, radius=1230, sobel_kernel_size=5):
        self.dir_path = dir_path
        self.d_a = d_a
        self.kernel_size = kernel_size
        self.cmap = plt.cm.get_cmap('gray_r', 5)
        self.sobel_kernel_size = sobel_kernel_size
        self.load_data()
        self.augmented_data = None


    def load_data(self):
        data_dir_path = os.path.join(self.dir_path, 'data')
        label_dir_path = os.path.join(self.dir_path, 'label')
        cluster_dir_path = os.path.join(self.dir_path, 'cluster')

        filename = 'rec_00400.tif'
        self.data = cv2.imread(os.path.join(data_dir_path, filename), cv2.IMREAD_GRAYSCALE)
        self.label = cv2.imread(os.path.join(label_dir_path, filename), cv2.IMREAD_GRAYSCALE)
        self.cluster = cv2.imread(os.path.join(cluster_dir_path, filename), cv2.IMREAD_GRAYSCALE)

    def image_rotate(self, image):
        angles = np.arange(0, 350 + self.d_a, self.d_a)
        r_img_list = [rotate(image, angle, mode='reflect', reshape=False, order=0) for angle in angles]
        return r_img_list

    def data_rotate(self, data):
        data_list = []
        for i in tqdm(range(data.shape[0])):
            data_j_list = []
            for j in range(data.shape[1]):
                data_j_list += self.image_rotate(data[i, j])
            data_list += data_j_list
        rotated_data = np.asarray(data_list)
        return rotated_data

    def prepare_image(self):
        data_blur, sobelx, sobely = Preparer.calculate(self.data,
                                                            sigma=0,
                                                            kernel_size=self.kernel_size,
                                                            sobel_kernel_size=5)
        return data_blur, sobelx, sobely

    def patchify_images(self, image):
        return patchify(image, (128, 128), step=128)

    def stack_images(self, cut_array, sobelx_array, sobely_array):
        return np.stack((cut_array, sobelx_array, sobely_array), axis=3)

    def split_data(self, X, y, test_size=0.65, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def save_data(self, save_dir, X_train, X_test, y_train, y_test):
        np.save(os.path.join(save_dir, 'x_train.npy'), X_train)
        np.save(os.path.join(save_dir, 'x_test.npy'), X_test)
        np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(save_dir, 'y_test.npy'), y_test)


    def augment_data(self, data_blur, sobelx, sobely):
        data_cut_array = self.patchify_images(data_blur)
        sobelx_cut_array = self.patchify_images(sobelx)
        sobely_cut_array = self.patchify_images(sobely)
        label_cut_array = self.patchify_images(self.label)

        rotated_cut_images = self.data_rotate(data_cut_array)
        rotated_sobelx_images = self.data_rotate(sobelx_cut_array)
        rotated_sobely_images = self.data_rotate(sobely_cut_array)
        rotated_label_images = self.data_rotate(label_cut_array)

        X = self.stack_images(rotated_cut_images, rotated_sobelx_images, rotated_sobely_images)
        rotated_label_images = np.expand_dims(rotated_label_images, axis=-1)

        X_train, X_test, y_train, y_test = self.split_data(X, rotated_label_images)

        save_dir = 'train_data/with_augmentation'
        self.save_data(save_dir, X_train, X_test, y_train, y_test)
        self.augmented_data = X_train, X_test, y_train, y_test