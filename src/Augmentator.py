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
    pass