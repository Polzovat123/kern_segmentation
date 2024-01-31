import PIL
import numpy as np
from numpy import typing as npt
from patchify import patchify
from scipy.ndimage import rotate
from tqdm import tqdm


class Augmentator:
    """Абстрактный функтор аугментации данных"""

    @classmethod
    def __patch_data(cls, data: npt.NDArray[np.float_], patch_size: int) -> npt.NDArray[np.float_]:
        print(data.shape)
        return patchify(data, (patch_size, patch_size), step=patch_size)

    @classmethod
    def __image_rotate(cls, image, d_a):
        angles = np.arange(0, 350 + d_a, d_a)
        r_img_list = []
        for angle in angles:
            r_img = rotate(image, angle, mode='reflect', reshape=False, order=0)
            r_img_list.append(r_img)
        return r_img_list

    @classmethod
    def __data_rotate(cls, data, d_a):
        data_list = []
        for i in tqdm(range(data.shape[0])):
            data_j_list = []
            for j in range(data.shape[1]):
                data_j_list += Augmentator.__image_rotate(data[i, j], d_a)
            data_list += data_j_list
        rotated_data = np.asarray(data_list)
        return rotated_data

    @classmethod
    def __add_noise(cls, data, sigma):
        noise = np.random.normal(0, sigma, data.shape)
        return data + noise

    @staticmethod
    def augment_data(data_blur: npt.NDArray[np.float_],
                     sobelx: npt.NDArray[np.float_],
                     sobely: npt.NDArray[np.float_],
                     patch_size: int = 128,
                     d_a: int = 10,
                     sigma: float = 0.5
                     ) -> (npt.NDArray[np.float_]):
        data_blur_patches = Augmentator.__patch_data(data_blur, patch_size)
        sobelx_patches = Augmentator.__patch_data(sobelx, patch_size)
        sobely_patches = Augmentator.__patch_data(sobely, patch_size)


        data_blur_rotated_patches = Augmentator.__data_rotate(data_blur_patches, d_a)
        sobelx_rotated_patches = Augmentator.__data_rotate(sobelx_patches, d_a)
        sobely_rotated_patches = Augmentator.__data_rotate(sobely_patches, d_a)

        X = np.stack((data_blur_rotated_patches, sobelx_rotated_patches, sobely_rotated_patches), axis=3)

        return X

    @staticmethod
    def __get_label_channels(label: npt.NDArray[np.float_]) -> list[npt.NDArray[np.float_]]:
        label_channels = []
        for i in range(3):
            label_channels.append(label[:, :, i])
        return label_channels
    @staticmethod
    def augment_label(label: npt.NDArray[np.float_],
                     patch_size: int = 64,
                     d_a: int = 10,
                     sigma: float = 0.1
                     ) -> (npt.NDArray[np.float_]):

        label_pathes = patchify(label, (patch_size, patch_size, 3), step=patch_size)

        label_rotated_pathes = Augmentator.__data_rotate(label_pathes, d_a)

        return label_rotated_pathes



