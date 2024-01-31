import cv2
import numpy as np
from numpy import typing as npt

class FurieGaussianSmoother:
    """Функтор сглаживания данных, основанный на процедуре фильтрации Фурье"""

    """
    def __init__(self, kernel_size: int = 5, sigma: float = 5):
        self._kernel_size = kernel_size
        self._sigma = sigma
        
    @property
    def kernel_size(self) -> int:
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, kernel_size: int) -> None:
        self._kernel_size = kernel_size

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float) -> None:
        self._sigma = sigma
    """

    @staticmethod
    def __calculate(data: npt.NDArray[np.float_], kernel_size: int, sigma: float) -> npt.NDArray[np.float_]:
        data_furie = np.fft.fft2(data)
        data_furie_shifted = np.fft.fftshift(data_furie)

        real_part = np.real(data_furie_shifted)
        imag_part = np.imag(data_furie_shifted)

        real_part_blur = cv2.GaussianBlur(real_part, (kernel_size, kernel_size), sigma)

        imag_part_blur = cv2.GaussianBlur(imag_part, (kernel_size, kernel_size), sigma)

        data_furie_blur = real_part_blur + 1j * imag_part_blur

        data_blur = np.fft.ifftshift(data_furie_blur)
        data_blur_result = np.fft.ifft2(data_blur)

        return np.abs(data_blur_result)

    @staticmethod
    def calculate(data: npt.NDArray[np.float_], kernel_size: int = 5, sigma: float = 5) -> npt.NDArray[np.float_]:
        return FurieGaussianSmoother.__calculate(data,kernel_size, sigma)

