from typing import Any, Sequence
import cv2
import numpy as np
from numpy import typing as npt, ndarray, dtype, generic


class TresholdCutter:
    """Функтор обрезки данных, основанный на процедуре отсечения по порогу"""
    _threshold = None

    def __init__(self, threshold: float = None):
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        self._threshold = threshold

    @staticmethod
    def get_auto_threshold(data: npt.NDArray[np.float_]) -> float:
        return cv2.threshold(data, 0, 1, cv2.THRESH_OTSU)[0]


    def __get_kernel_contours(self, data: npt.NDArray[np.float_]) -> Sequence[
        ndarray | ndarray[Any, dtype[generic | generic]] | Any]:
        if self._threshold is None:
            self._threshold = self.get_auto_threshold(data)

        _, thresh_img = cv2.threshold(data, self._threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def __get_ellipse_contour(self, data: npt.NDArray[np.float_], contours: Sequence[
        ndarray | ndarray[Any, dtype[generic | generic]] | Any]) -> ndarray | ndarray[Any, dtype[generic | generic]] | Any:
        max = 0
        sel_countour = None
        for countour in contours:
            if countour.shape[0] > max:
                sel_countour = countour
                max = countour.shape[0]

        ellipse_contour = sel_countour.reshape((-1, 1, 2))

        return ellipse_contour

    def cut_data(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        contours_mask = self.__get_kernel_contours(data)
        ellipse_contour = self.__get_ellipse_contour(data, contours_mask)


        mask_ellipse = np.zeros_like(data, dtype=np.uint8)
        cv2.drawContours(mask_ellipse, [ellipse_contour], -1, (255, 255, 255), thickness=cv2.FILLED)


        mask_ellipse_inv = ~mask_ellipse

        data = data.astype(float)
        data[mask_ellipse_inv == 255.0] = np.nan
        return data



