# sign detection on images

# some ideas come from here https://github.com/ahmetozlu/signature_extractor

import os
import cv2 as cv
from skimage import morphology, color, measure
import pandas as pd
import matplotlib.pyplot as plt

from crop import crop

# img = cv.imread('data/my/4.tif')
# let's cut the top 25% of image
# usually it doesn't have any signatures
# img = crop_top(img)
# cv.namedWindow('cropped', cv.WINDOW_NORMAL)
# cv.imshow('cropped', img)
# cv.waitKey()
# ####### TEST SECTION ########## #

# contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# img = morphology.erosion(img)
# cv.namedWindow('erosion', cv.WINDOW_NORMAL)
# cv.imshow("erosion", img_eroded)
# # do not close a window by mouse, just hit a spacebar or any number/letter key
# cv.waitKey()
# # cv.destroyWindow("fit")
# raise SystemExit
# ####### END of TEST  ########## #

# what = img > img.mean() # this convert to array with true / false values
"""
blobs = measure.label(img, background=1, connectivity=1)
blobs_props = measure.regionprops(blobs)
#for blob in blobs_props:
#     print(blob.area)
# giants = [x for x in blobs_props if x.area > 3000]
giants = [x for x in blobs_props if x.area > 4000]
bboxes = [x.bbox for x in giants]
areas = [x.area for x in giants]
"""

# print(list(zip(areas, bboxes)))
# print(len(giants))

# it's already binary
# img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
# cv.namedWindow('fit', cv.WINDOW_NORMAL)
# cv.imshow("fit", img)
# do not close a window by mouse, just hit a spacebar or any number/letter key
# cv.waitKey()
# cv.destroyWindow("fit")


# print(sorted(area))
# giants = sorted(area)[-10:]
# print(giants)
# what_over = color.label2rgb(blobs_labels)

"""
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(img)
counter = 0
proper_giants = []
for box in bboxes:
    ymin, xmin, a, ymax, xmax, b = box
    if ymax - ymin < 42 or ymax - ymin > 350 or xmax - xmin < 68:
        continue
    print(box)
    by = (ymin, ymax, ymax, ymin, ymin)
    bx = (xmin, xmin, xmax, xmax, xmin)
    ax.plot(bx, by, '-b', linewidth=2, marker='x')
    counter += 1
print(counter)    
ax.set_axis_off()
plt.tight_layout()
plt.show()
"""

# ############ TEST PROBLEMATIC ############# #
problematic = ['0370ca51b827c0f72a8f6b6a83e4411a.tif', '02701be3fc048a3b7e2ce7a4b2a05aad.tif', '1b5b26eea70d2bed49b6a17d456e5662.tif', '2af4976206c338874b12fd33ec18aceb_32.tif',
'32b1fc5cbe73d5465ef6782e14cb8d1c.tif', '2b39a63db00417e8838bcc6256e903a1_1.tif', '3598b981fd75d09523dd049614fab8f1.tif',
'35cabb4a166bfca8b4a49f8d5bbf74f2_1.tif']

# raise SystemExit
# ############## TEST ALL PART ############## #

folder_path = os.path.abspath('data/test')
all_files = os.listdir(folder_path)
all_results = []
all_names = []
for file in all_files:
# for file in problematic[:1]:
    print(file)
    all_names.append(file[:-4])
    all_results.append(0)
    img = cv.imread(f'data/test/{file}')
    # let's cut the top 25% of image
    # usually it doesn't have any signatures
    img = crop(img, percentage=20, log=0)
    img = morphology.erosion(img)
    # img = morphology.erosion(img)

    blobs = measure.label(img, background=1, connectivity=1)
    blobs_props = measure.regionprops(blobs)
    # for blob in blobs_props:
    #     print(blob.area)
    giants = [x for x in blobs_props if x.area > 2000]
    # bboxes = [x.bbox for x in giants]
    # areas = [x.area for x in giants]
    # area_convex = [x.area_convex for x in giants]

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(img)
    counter = 0

    for giant in giants:
        box = giant.bbox
        area = giant.area
        ratio = giant.extent
        ymin, xmin, a, ymax, xmax, b = box
        if ymax - ymin < 52 or ymax - ymin > 350 or xmax - xmin < 22 or ratio > 0.3:
            continue
        # print('found')
        all_results[-1] = 1
        # print(all_results)
        # print(box, area, ratio)
        # by = (ymin, ymax, ymax, ymin, ymin)
        # bx = (xmin, xmin, xmax, xmax, xmin)
        # ax.plot(bx, by, '-b', linewidth=2, marker='x')
        counter += 1
    print(counter)    
    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()
print(all_results)
df = pd.DataFrame()
df['Id'] = all_names
df['Expected'] = all_results
df.to_csv('test_results.csv', index=False)