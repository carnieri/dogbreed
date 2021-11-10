"""Remove totally black images from dataset"""

import os
from pathlib import Path

import cv2
import numpy as np

path = Path("dogs/train")
for p in path.glob("**/*.jpg"):
    img = cv2.imread(str(p))
    if np.alltrue(img == 0):
        print(f"{p} is totally black, deleting")
        os.remove(p)
