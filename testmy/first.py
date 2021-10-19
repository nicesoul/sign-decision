# sign detection on images

# some ideas come from here https://github.com/ahmetozlu/signature_extractor

# import os
import cv2 as cv
from skimage import morphology, color, measure
import matplotlib.pyplot as plt

img = cv.imread('../data/my/2.tif')


# what = img > img.mean() # this convert to array with true / false values

blobs_labels = measure.label(img, background=1)
blobs_props = measure.regionprops(blobs_labels)
#for blob in blobs_props:
#     print(blob.area)
giants = [x for x in blobs_props if x.area > 3000]
bboxes = [x.bbox for x in giants]
areas = [x.area for x in giants]


print(list(zip(areas, bboxes)))
print(len(giants))

# it's already binary
# img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
# cv.namedWindow('fit', cv.WINDOW_NORMAL)
# cv.imshow("fit", img)
# do not close a window by mouse, just hit a spacebar or any number/letter key
# cv.waitKey()
# cv.destroyWindow("fit")


# print(sorted(area))
#giants = sorted(area)[-10:]
#print(giants)
# what_over = color.label2rgb(blobs_labels)


fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(img)
for box in bboxes:
    ymin, xmin, a, ymax, xmax, b = box
    if ymax - ymin < 25:
        continue
    by = (ymin, ymax, ymax, ymin, ymin)
    bx = (xmin, xmin, xmax, xmax, xmin)
    ax.plot(bx, by, '-b', linewidth=2.5)

ax.set_axis_off()
plt.tight_layout()
plt.show()


# ############ Careful, some borrowed code below ############ #

