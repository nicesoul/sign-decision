# sign detection on images

# some ideas come from here https://github.com/ahmetozlu/signature_extractor

import os
import cv2 as cv
from skimage import morphology, color, measure
import pandas as pd
import matplotlib.pyplot as plt

from crop import crop

# img = cv.imread('data/my/4.tif')
# let's cut the top 20% of image and some stuff from the sides
# usually it doesn't have any signatures
# img = crop_(img)
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

# ############ TEST PROBLEMATIC ############# #
problematic = ['0370ca51b827c0f72a8f6b6a83e4411a.tif', '02701be3fc048a3b7e2ce7a4b2a05aad.tif', '1b5b26eea70d2bed49b6a17d456e5662.tif',
               '2af4976206c338874b12fd33ec18aceb_32.tif',
               '32b1fc5cbe73d5465ef6782e14cb8d1c.tif', '2b39a63db00417e8838bcc6256e903a1_1.tif', '3598b981fd75d09523dd049614fab8f1.tif',
               '35cabb4a166bfca8b4a49f8d5bbf74f2_1.tif']
test_ratio = ['1b6d0840c3277fe4ac4215b5bc8aecfd.tif', '23da1f0aae50d9e2b044b599dcaea38e_2.tif', '7f19a1ea9dbcdf99a9c78a7fb65d3c9b.tif',
              'bd017a684aa93257d7c79fedde4d24e7_3.tif', 'c98a06bc768ed92afea8be4a67c26911_2.tif', 'cb07b3779ee43a9bb7eecb322a6b4160_2.tif',
              'ce457dce276f4915180fef71c775162e_3.tif', 'd4dfbdddc3e02c1e4891773f0b26f73e.tif', '2af4976206c338874b12fd33ec18aceb_32.tif',
              'ad729ba5891d6199f01d12d59086bb86_2.tif', 'ad729ba5891d6199f01d12d59086bb86_3.tif', 'c816e5457fc313c27bd38ebc1658b4b8_2.tif',
              'e1c39b2cd75cbe3e140791382085cd6b_2.tif', 'c29f7d2b95144642cea08a4f4455efe9_2.tif', 'c29f7d2b95144642cea08a4f4455efe9_3.tif',
              'cf250e5f053cd65aa930c7d76f264d11_1.tif', '1b5b26eea70d2bed49b6a17d456e5662.tif']
check_crop = '0148e4778e6d8b22a8980bf4fc89351b_27.tif 0171cd73c7bdbf48ccfd0c21804f9ced_2.tif 035b1329050bb966ac4c9a6af5b2df3b_2.tif \
051781786f9fb429f4cc6508ac8e0676_2.tif 081a5d93952b57ecde40bc02b8b70e67_1.tif 0dab8c0f23691b43509fc7594d6ffdcd_2.tif \
0f28ae18bbe47822f72b59ec831a4aaa_2.tif 30deb3462a2419392e1ffee3d002f313_3.tif 30deb3462a2419392e1ffee3d002f313_4.tif \
329a4f816a4d943b55136d3e0f3e82d2.tif 3df9b5edbeb553e788f4bad27c562783_2.tif 7747787780814a71eab4c37f573115c2_3.tif \
79d87e9469873c5672ad32028be45891.tif 7db6b19dd652e0a44e0948a1250a7429_5.tif 8ba85d52f5b2a1eb9061c90f656fcc4f.tif \
8e20c5b100299349efd0339019392688_1.tif 96c5c51c7f218b0dacd8685559672232_2.tif a26e1c8a414da8d4771588f32b9b0b2a.tif'.split()
check_lines = '0f28ae18bbe47822f72b59ec831a4aaa_1.tif 0f6bc860c0dbac06933e2dbae943b850_3.tif 28282c85db1849b9a921fd4dacf8e189_1.tif 2901d78cd3470b6283bfdc8ccb41538a_12.tif 30deb3462a2419392e1ffee3d002f313_5.tif 3164d35d0b585df7f818684d9601308a_1.tif 3cf16ae1c4b21b53eba598ede578222a_14.tif 3cf16ae1c4b21b53eba598ede578222a_3.tif 3cf16ae1c4b21b53eba598ede578222a_4.tif 3cf16ae1c4b21b53eba598ede578222a_9.tif 3d84c7fde5e0ff7963533b3f14d2ec22_2.tif 3df9b5edbeb553e788f4bad27c562783_1.tif 4106cb065758ce92428b34f0d15e4848_17.tif 43a6dab7fbd361a4870a66916959c118_1.tif 4532b01ca7c1f765d0fbbcf874f18662_1.tif 488c643b183ca48752b476730c9fe019_2.tif 4be855f1ee97395aa3b1d7173491a37f.tif 52a641fd852639daaa45556a3d7f3a55_4.tif 52a641fd852639daaa45556a3d7f3a55_7.tif 5b3e4de6da2a3165c4ee76b543f205ea_1.tif 72590e77e4afbde1c558f4073f669093_1.tif 74108176fee8e408092ac23b866e8217_1.tif 74108176fee8e408092ac23b866e8217_2.tif 7679ff1c8fc5239a90e06d16b03d36a8_3.tif 834116f765417fb3895d098d6d2ecefb_1.tif 84b1e44dd65cefe2811e7b9a34c153dc_1.tif 87fa366a5da6c35ccbca7863b917d527_2.tif 8c23e586a79dd05dcaa130aced55d2ef_2.tif 8c23e586a79dd05dcaa130aced55d2ef_3.tif 96c9d24f6ccf0899d15c882e062ffc83_1.tif 96c9d24f6ccf0899d15c882e062ffc83_4.tif 996a9d6db1d796eeeee39303adfb252d_2.tif 9df6c61d454c12e31a006df97b63dc57_2.tif a2728eb52e274bb2de91165ccdd90102_1.tif a9d9430057a2028da698fe3326d631fb_2.tif aa18781ad618f6aae949142d51d13b21_5.tif ad729ba5891d6199f01d12d59086bb86_4.tif ad729ba5891d6199f01d12d59086bb86_5.tif ad729ba5891d6199f01d12d59086bb86_6.tif bd017a684aa93257d7c79fedde4d24e7_1.tif bd017a684aa93257d7c79fedde4d24e7_2.tif 04876c4333a785c98693314e2b334bba_2.tif 051781786f9fb429f4cc6508ac8e0676_1.tif 0533a1e934bd69947b247d0e858d854f_2.tif 14173ae97a1750ae36222cdfb90772d5_2.tif 23f0c15343fcf9b94ec1f7d02a5b5524_2.tif 2889b4e52a72cda7c333f195fb1a734b_2.tif 2af4976206c338874b12fd33ec18aceb_10.tif 2af4976206c338874b12fd33ec18aceb_15.tif 2af4976206c338874b12fd33ec18aceb_30.tif 3cf16ae1c4b21b53eba598ede578222a_15.tif 4179340310a187ea342a21b49be7d3a1_2.tif 5740f241edc8a888a1004e236332ecbe_3.tif 65f1634f6d208e718df532717bbdc62e_2.tif 65f1634f6d208e718df532717bbdc62e_3.tif 6a7d0a04850b7198951ae7325caccbdd_3.tif 75afebf5f040177e9b1e35d8119a04c3_1.tif 81e513afffd3423a3e34efd26928a11e_4.tif 853bb94f764223703e565c8526f63548_2.tif 996a9d6db1d796eeeee39303adfb252d_3.tif 9df6c61d454c12e31a006df97b63dc57_3.tif a4728fe872019967f10b40f76f8917de_2.tif a47af9d24e4e7f94db98a1e438135567_2.tif b70372c7ddd50608ac1a0d3815e0fba5_1.tif be27806774c6c259e470bec61741a262_2.tif 107de8cc48fea1bc14862ed9a470d42a_2.tif 41263e2b8a0c255da8ef84cd9b9bcca0_2.tif 4b520da4e68716484a7374a42bf7e2a9_1.tif 8a807c1b9885f8408875a6650c6ffe13_1.tif 8b2e30ff3b7c51c0781bf3e2b4078927_1.tif 9e49e47465746567875e58c37f09decd_1.tif b562467b40bc3ebb9f88187ddefa2b14_1.tif'.split()
# raise SystemExit
# ############## TEST ALL PART ############## #

folder_path = os.path.abspath('data/test')
# folder_path = os.path.abspath('data/train')
all_files = os.listdir(folder_path)
all_results = []
all_names = []
files_counter = 0
start = 0
# for file in problematic[:1]:
# for file in test_ratio:
# for file in all_files[start:]:
for file in all_files:
    print(f'files counter is {start+files_counter}')
    # debugging counter
    files_counter += 1
    print(file)
    # creating 2 lists with names and values defaulting to 0
    all_names.append(file[:-4])
    all_results.append(0)
    img = cv.imread(f'data/test/{file}')
    # img = cv.imread(f'data/train/{file}')
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

    # bboxes = [x.bbox for x in giants]
    # areas = [x.area for x in giants]
    # area_convex = [x.area_convex for x in giants]

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(img)
    # counter for succesful giant boxes => possible signatures
    counter = 0
    # max width of signature line
    width_max = img.shape[1] * 0.6
    sign_min = img.shape[0] * 0.03
    sign_max = img.shape[0] * 0.3
    for giant in giants:
        box = giant.bbox
        area = giant.area
        ratio = giant.extent
        ymin, xmin, a, ymax, xmax, b = box
        if ymax - ymin < sign_min or ymax - ymin > sign_max or xmax - xmin < sign_min or xmax - xmin > width_max or ratio < 0.05 or ratio > 0.33:
            continue
        all_results[-1] = 1
    #     print(box, area, ratio)
    #     by = (ymin, ymax, ymax, ymin, ymin)
    #     bx = (xmin, xmin, xmax, xmax, xmin)
    #     ax.plot(bx, by, '-b', linewidth=2, marker='x')
    #     counter += 1
    # print(counter)
    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()
# raise SystemExit
print(all_results)
df = pd.DataFrame()
df['Id'] = all_names
df['Expected'] = all_results
df.to_csv('test_results.csv', index=False)
