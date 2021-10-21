# util to crop the image
# initially it was simple just to cut the top
# now it's becoming advance, seems to be working fine (21.10.2021)

import numpy as np

CUT_definition = 'Provide a proper <cut> argument (float or int) in crop.py\n\
<cut> argument between 0 and 1 is percentage, more than 1 is number of pixels\n\
e.g. cut=0.42 is 42% and cut=33 is 33 pixels'

SIDE_definition = "Provide a proper <side> argument (str or list) in crop.py\n\
possible <side> values are 't' for top, \
'b' for bottom, 'l' for left, and 'r' for right. \
You may combine sides -> side='tb' to cut from top and bottom.\n"


def crop(img: np.ndarray, side: str = 't', cut: float = 0.2, log: bool = 0) -> np.ndarray:
    """
    util to crop the image.
    defaults to 20 % from the top.\n
    possible <side> values are 't' for top,
    'b' for bottom, 'l' for left, and 'r' for right.
    You may combine sides -> side='tb' to cut from top and bottom.\n
    <cut> argument between 0 and 1 is percentage, more than 1 is number of pixels
    e.g. cut=0.42 is 42% and cut=33 is 33 pixels.\n
    log=1 to print dimensions.
    """
    if not isinstance(side, (str, list)):
        raise SystemExit(SIDE_definition)
    if not isinstance(cut, (float, int)):
        raise SystemExit(CUT_definition)
    if 0 < cut < 1:
        method = 'percent'
    elif cut > 1:
        method = 'pixel'
    else:
        raise SystemExit(CUT_definition)
    top, left = 0, 0
    bottom = img.shape[0]
    right = img.shape[1]
    if cut > bottom or cut > right:
        raise SystemExit('Arguments are bigger than image dimensions.\
    Check value of <cut> in crop.py')
    if 't' in side:
        top = calculate(bottom, method, cut)
    if 'b' in side:
        bottom = bottom - calculate(bottom, method, cut)
    if 'l' in side:
        left = calculate(right, method, cut)
    if 'r' in side:
        right = right - calculate(right, method, cut)
    if log:
        print(f'arguments provided: side={side}, cut={cut}, log={log}')
        print(f"original dimensions are {img.shape}")
        print('top, bottom, left, right: ', top, bottom, left, right)
        print(F"cropped dimensions are {img[top:bottom, left:right].shape}")
    return img[top:bottom, left:right]


def calculate(side, method, cut) -> int:
    """
    small util for crop function.
    calculates the new side value.
    """
    if method == 'percent':
        return int(side * cut)
    if method == 'pixel':
        return int(cut)
