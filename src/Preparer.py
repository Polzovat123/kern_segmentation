import cv2
import numpy as np
from numpy import typing as npt

from src.utils.Cutter import TresholdCutter


class Preparer:
    """Функтор препроцессинга данных"""

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

        data_blur = cv2.GaussianBlur(data, (kernel_size, kernel_size), sigma)
        cutted_data = _trashold_cutter.cut_data(data_blur)
        scaled_data = Preparer.__std_scaler(cutted_data)

        sobelx = cv2.Sobel(src=scaled_data, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=sobel_kernel_size)
        sobely = cv2.Sobel(src=scaled_data, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=sobel_kernel_size)

        return cutted_data, sobelx, sobely