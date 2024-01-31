from abc import abstractmethod, ABCMeta

import PIL
import cv2
import numpy as np
from numpy import typing as npt

from src.utils.Smoother import FourierGaussianSmoother
from src.utils.Cutter import TresholdCutter


class Preparer:
    """Функтор препроцессинга данных"""

    @staticmethod
    def __get_nan_mask(data: npt.NDArray[np.float_]) -> npt.NDArray[np.bool_]:
        return np.isnan(data)

    @staticmethod
    def __fill_by_nan(data: npt.NDArray[np.float_], mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.float_]:
        data_nan = np.zeros_like(data, dtype=float)
        data_nan[mask] = data[mask]
        data_nan[~mask] = np.nan
        return data_nan

    @staticmethod
    def __fill_nan(data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        mask = np.isnan(data)
        data_ = np.zeros_like(data, dtype=float)
        data_[mask] = np.mean(data[~mask])
        data_[~mask] = data[~mask]
        return data_

    @staticmethod
    def __std_scaler(my_data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        mean = np.nanmean(my_data)
        std = np.nanstd(my_data)
        standardized_data = (my_data - mean) / std
        return standardized_data

    @staticmethod
    def calculate(data: npt.NDArray[np.float_],
                    sigma: float = 0, kernel_size: int = 3,  sobel_kernel_size: int = 5) -> (
    npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]):
        _trashold_cutter = TresholdCutter()
        #_furie_gaussian = FourierGaussianSmoother() пока не работает

        #data_blur = cv2.boxFilter(data, -1, (kernel_size, kernel_size), normalize=True)
        data_blur = cv2.GaussianBlur(data, (kernel_size, kernel_size), sigma)
        cutted_data = _trashold_cutter.cut_data(data_blur)
        scaled_data = Preparer.__std_scaler(cutted_data)

        sobelx = cv2.Sobel(src=scaled_data, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=sobel_kernel_size)
        sobely = cv2.Sobel(src=scaled_data, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=sobel_kernel_size)

        return cutted_data, sobelx, sobely