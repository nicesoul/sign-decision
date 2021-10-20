# util to crop the image
# initially it was simple just to cut the top
# now it's becoming advance, let me finish it later (20.10.2021)

import cv2 as cv
import numpy as np


def crop(img: np.ndarray, side: str = 't', percentage: int = 25, px: int = 20, log: bool = 0) -> np.ndarray:
    """
    WARNING: CUTS 20 pixels from bottom and percentage from the Top\n
    util to crop the image.
    defaults to 25 % from the top\n
    possible <side> values are 't' for top,
    'b' for bottom, 'l' for left, and 'r' for right.
    you may combine sides -> side='tb'\n 
    provide <percentage> argument between 1 and 100
    or <px> argument as amount of pixels
    log=1 to print dimensions
    """
    new_y = int(img.shape[0] * percentage / 100)
    if log:
        print(f"original dimensions are {img.shape}")
        print(F"cropped dimensions are {img[new_y:-px,:].shape}")
    return img[new_y:-px, :]


# img = cv.imread('../data/my/1.tif')
# crop(img, log=1)
