import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image

import numpy as np
import matplotlib.pyplot as plt
import cv2
from patch_extractor import normalize_staining
from numpy import linalg

from math import sqrt

import matplotlib.pyplot as plt

from skimage import data
from skimage.color import separate_stains, rgb2grey
from skimage.exposure import rescale_intensity
from skimage.feature import blob_dog
from skimage.transform import rotate

FILE = 'data/train/Normal/t51.tif'

from skimage import data
from patch_extractor import extract_patches_overlap

if __name__ == "__main__":
    # img1 = np.load('data/features_final/train/b2_0.npy')
    # img2 = np.load('data/features_final/train/b2_2.npy')
    # img3 = np.load('data/features_final/train/b2_7_951.npy')
    # img4 = np.load('data/features_final/train/b2_7.npy')
    # print(img1)
    # print(img2)
    # print(img3)
    # print(img4)
    # plt.imshow(img_array[0][11], cmap='gray')
    # plt.show()

    # one_image = np.arange(144).reshape((12,12))
    # print(one_image)

    # img = cv2.imread(FILE)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_norm = normalize_staining(img)

    # patches = image.extract_patches_2d(img,(512,512), 12)
    # print(patches.shape)
    # f, axarr = plt.subplots(2,2)
    # axarr[0,0].imshow(patches[0])
    # axarr[0,1].imshow(patches[1])
    # axarr[1,0].imshow(patches[2])
    # axarr[1,1].imshow(patches[3])
    # plt.show()

    # for patch in patches:
    #     print('----')
    #     print(patch)

    # plt.imshow(img_norm, cmap='gray')
    # plt.show()
    
    # rotate_seed = np.random.randint(-10,10)
    # print(rotate_seed)
    # image_with_rotation = rotate(img_norm, rotate_seed)

    # plt.imshow(image_with_rotation, cmap='gray')
    # plt.show()
    # for i in [0,1,2,3,4,5,6,7]:
    #     aug_seed = np.random.randint(0,4)
    #     print('seed: ', aug_seed)
    #     if(aug_seed == 0):
    #         print('donothing')
    #     elif(aug_seed == 1):
    #         rotate_seed = np.random.randint(-10,10)
    #         print('rotate: ' ,rotate_seed)
    #     elif(aug_seed == 2):
    #         print('flip')
    #     elif(aug_seed == 3):
    #         rotate_seed = np.random.randint(-10,10)
    #         print('rotate: ' ,rotate_seed)
    #         print('flip')
    

    # rotate_seed = np.random.randint(-10,10)
    
    # patch_rotate = img_norm.copy()
    # patch_rotate = rotate(patch_rotate, rotate_seed)
    # plt.imshow(patch_rotate, cmap='gray')
    # plt.show()

    # patch_flip = img_norm.copy()
    # patch_flip = patch_flip[:, ::-1]
    # plt.imshow(patch_flip, cmap='gray')
    # plt.show()

    # patches = extract_patches_overlap(img, (512,512), 0.5)
    # print(patches.shape)
    # # for patch in patches:

    
    # f, axarr = plt.subplots(2,2)
    # axarr[0,0].imshow(patches[0])
    # axarr[0,1].imshow(patches[1])
    # axarr[1,0].imshow(patches[2])
    # axarr[1,1].imshow(patches[3])

    # plt.show()

    # x = np.load('data/features/train/b0_0.npy')
    # print(x)

    x = np.load('data/features_one_class/train/b12_he2_0.npy')
    print(x)
    print(x.shape)