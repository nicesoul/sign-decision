# sign detection on images
# sorry, guys, no time to make it a function
# or even finish documentation

# some ideas come from here https://github.com/ahmetozlu/signature_extractor

import os
import cv2 as cv
from skimage import morphology, color, measure
import pandas as pd
import matplotlib.pyplot as plt

from crop import crop

folder_path = os.path.abspath('data/test')
all_files = os.listdir(folder_path)

all_results = []
all_names = []
files_counter = 0
# continue where you stopped your checks
start = 0
for file in all_files[start:]:
    print(f'files counter is {start+files_counter}')
    # debugging counter
    files_counter += 1
    print(file)
    # creating 2 lists with names and values defaulting to 0
    all_names.append(file[:-4])
    all_results.append(0)

    # changes to read as 1 dimension image
    # not implemented, RGB works better even for grayscale images - I don't know why :)
    # img = cv.imread(f'data/test/{file}', cv.IMREAD_ANYDEPTH)
    # check additional flags here
    # https://docs.opencv.org/4.5.3/d8/d6a/group__imgcodecs__flags.html#ga61d9b0126a3e57d9277ac48327799c80

    img = cv.imread(f'{folder_path}/{file}')
    # let's cut the top 20% of image
    # usually it doesn't have any signatures
    img = crop(img, side='t', cut=0.2, log=0)
    # and 5% from the right, bottom, left sides
    img = crop(img, side='robolo', cut=0.05, log=0)

    # here we increase the thickness of lines in image
    # trying to connect different signatures parts into one line
    # because signarutes are often weak with missing pixels
    img = morphology.erosion(img)

    # scikit-image method to group connected pixels
    blobs = measure.label(img, background=1, connectivity=1)
    # and this one to get properties of those groups
    blobs_props = measure.regionprops(blobs)
    # ignore small groups
    giants = [x for x in blobs_props if x.area > 2000]

    # counter for succesful giant boxes => possible signatures
    counter = 0
    # dimensions of signature
    height_min = int(img.shape[0] * 0.02)
    height_max = int(img.shape[0] * 0.5)
    # max width of signature line
    width_max = int(img.shape[1] * 0.5)
    width_min = int(img.shape[1] * 0.01)
    height_min_px = 52
    height_max_px = 480
    width_min_px = 22

    fig, ax = plt.subplots(figsize=(10, 5))

    for giant in giants:
        box = giant.bbox
        area = giant.area
        ratio = giant.extent
        # if grayscale then 4 values like
        # ymin, xmin, ymax, xmax = box
        # if default then 6 values
        ymin, xmin, ab, ymax, xmax, aa = box
        height = ymax - ymin
        width = xmax - xmin
        vertical_max = (height) / (width)

        # 73 % value line is looks like this
        # if height < 52 or height > 350 or width < 22 or ratio > 0.3:
        if height < height_min or height < height_min_px or height > height_max or height > height_max_px or vertical_max > 3 or \
           width < width_min or width < width_min_px or width > width_max or ratio < 0.048 or ratio > 0.298:
            continue
        all_results[-1] = 1

        by = (ymin, ymax, ymax, ymin, ymin)
        bx = (xmin, xmin, xmax, xmax, xmin)
        ax.plot(bx, by, '-g', linewidth=2, marker='x')
        counter += 1
    print(counter)
    ax.set_axis_off()
    plt.tight_layout()
    plt.imshow(img, cmap='Greys_r')
    plt.show()

# if you want a CSV, uncomment exception below
raise SystemExit

print(all_results)
df = pd.DataFrame()
df['Id'] = all_names
df['Expected'] = all_results
df.to_csv('your_results.csv', index=False)
