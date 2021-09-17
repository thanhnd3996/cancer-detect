import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image

import matplotlib.pyplot as plt
import cv2
from numpy import linalg

from math import sqrt

import matplotlib.pyplot as plt

from skimage import data
from skimage.color import separate_stains, rgb2grey
from skimage.exposure import rescale_intensity
from skimage.feature import blob_dog

FILE = 'data/train/Normal/t9.tif'

from skimage import data

def nuclei_count(img):
    #Color deconvolution
    #Hematoxylin(0) perm red(1) & DAB(2)

    rgb_from_hrd = np.array([[0.644, 0.710, 0.285],
                         [0.0326, 0.873, 0.487],
                         [0.270, 0.562, 0.781]])
    hrd_from_rgb = linalg.inv(rgb_from_hrd)

    ihc_hrd = separate_stains(img, hrd_from_rgb)

    '''Hematoxylin'''
    #Rescale signals
    #[:, :, 012 color]
    hema_rescale = rescale_intensity(ihc_hrd[:, :, 0], out_range=(0, 1))

    hema_array = np.dstack((np.zeros_like(hema_rescale), hema_rescale, hema_rescale))

    #Blob detection
    image2dH = rgb2grey(hema_array)

    blobs_DoG_Hema = blob_dog(image2dH, min_sigma=10, max_sigma=15, threshold=.25, overlap=0.9)
    blobs_DoG_Hema[:, 2] = blobs_DoG_Hema[:, 2] * sqrt(2)

    # num_blobsD = len(blobs_DoG_DAB)
    num_blobsH = len(blobs_DoG_Hema)
    return num_blobsH

def normalize_staining(img):
    """
    Adopted from "Classification of breast cancer histology images using Convolutional Neural Networks",
    Teresa Araújo , Guilherme Aresta, Eduardo Castro, José Rouco, Paulo Aguiar, Catarina Eloy, António Polónia,
    Aurélio Campilho. https://doi.org/10.1371/journal.pone.0177544

    Performs staining normalization.

    # Arguments
        img: Numpy image array.
    # Returns
        Normalized Numpy image array.
    """
    Io = 240
    beta = 0.15
    alpha = 1
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])

    h, w, c = img.shape
    img = img.reshape(h * w, c)
    OD = -np.log((img.astype("uint16") + 1) / Io)
    ODhat = OD[(OD >= beta).all(axis=1)]
    W, V = np.linalg.eig(np.cov(ODhat, rowvar=False))

    Vec = -V.T[:2][::-1].T  # desnecessario o sinal negativo
    That = np.dot(ODhat, Vec)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if vMin[0] > vMax[0]:
        HE = np.array([vMin, vMax])
    else:
        HE = np.array([vMax, vMin])

    HE = HE.T
    Y = OD.reshape(h * w, c).T

    C = np.linalg.lstsq(HE, Y)
    maxC = np.percentile(C[0], 99, axis=1)

    C = C[0] / maxC[:, None]
    C = C * maxCRef[:, None]
    Inorm = Io * np.exp(-np.dot(HERef, C))
    Inorm = Inorm.T.reshape(h, w, c).clip(0, 255).astype("uint8")

    return Inorm


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
    img = cv2.imread(FILE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = normalize_staining(img)
    patches = image.extract_patches_2d(img,(512,512), 150)
    
    # nuclei_num = nuclei_count(patches[0])
    # print('Number of nuclei: ' ,  nuclei_num)
    # plt.imshow(patches[0], cmap='gray')
    # plt.show()

    for index, patch in enumerate(patches):
        nuclei_num = nuclei_count(patch)
        print('Number of nuclei on patch_' ,index, '_',  nuclei_num)
        if(nuclei_num <= 20):
            plt.imshow(patch, cmap='gray')
            plt.show()
    