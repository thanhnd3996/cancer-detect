#!/usr/bin/env python3

import numpy as np
from numpy.random import randint
import cv2
from sklearn.feature_extraction import image as patch_ex
from matplotlib import pyplot as plt
from os.path import basename, join, exists
from os import makedirs
import sys
from nuclei_detect import nuclei_count
from skimage.transform import rotate
import shutil

PATCH_SIZE = 256
# PATCH_SIZE = 512
NUM_PER_PATCH = 20
# ROTATE_DEGREE = 90
# NUCLEI_THRESHOLD = 10

# INPUT_DIR = 'data/train'
INPUT_DIR = 'data/test'

# PREPROCESS_DIR = 'data/preprocessed/train'
# PREPROCESS_DIR = 'data/preprocessed/train_overlap'
PREPROCESS_DIR = 'data/preprocessed/test'

# CLASSES = ['Normal', 'Benign', 'Invasive', 'In Situ']
# CLASSES = ['Benign', 'Invasive', 'In Situ']
# CLASSES = ['Normal']
CLASSES = ['test']

PATCH_TYPE = 'Overlap'


# PATCH_TYPE = 'Random'

def recursive_glob(root_dir, file_template="*.tif"):
    """Traverse directory recursively. Starting with Python version 3.5, the glob module supports the "**" directive"""

    if sys.version_info[0] * 10 + sys.version_info[1] < 35:
        import fnmatch
        import os
        matches = []
        for root, dirnames, filenames in os.walk(root_dir):
            for filename in fnmatch.filter(filenames, file_template):
                matches.append(os.path.join(root, filename))
        return matches
    else:
        import glob
        return glob.glob(root_dir + "/**/" + file_template, recursive=True)


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


def hematoxylin_eosin_aug(img, low=0.7, high=1.3, seed=None):
    """
    "Quantification of histochemical staining by color deconvolution"
    Arnout C. Ruifrok, Ph.D. and Dennis A. Johnston, Ph.D.
    http://www.math-info.univ-paris5.fr/~lomn/Data/2017/Color/Quantification_of_histochemical_staining.pdf

    Performs random hematoxylin-eosin augmentation

    # Arguments
        img: Numpy image array.
        low: Low boundary for augmentation multiplier
        high: High boundary for augmentation multiplier
    # Returns
        Augmented Numpy image array.
    """
    D = np.array([[1.88, -0.07, -0.60],
                  [-1.02, 1.13, -0.48],
                  [-0.55, -0.13, 1.57]])
    M = np.array([[0.65, 0.70, 0.29],
                  [0.07, 0.99, 0.11],
                  [0.27, 0.57, 0.78]])
    Io = 240

    h, w, c = img.shape
    OD = -np.log10((img.astype("uint16") + 1) / Io)
    C = np.dot(D, OD.reshape(h * w, c).T).T
    r = np.ones(3)
    r[:2] = np.random.RandomState(seed).uniform(low=low, high=high, size=2)
    img_aug = np.dot(C, M) * r

    img_aug = Io * np.exp(-img_aug * np.log(10)) - 1
    img_aug = img_aug.reshape(h, w, c).clip(0, 255).astype("uint8")
    return img_aug


def extract_patches_overlap(image, patchshape, overlap_allowed=0.5, cropvalue=None,
                            crop_fraction_allowed=0.1):
    """
    Given an image, extract patches of a given shape with a certain
    amount of allowed overlap between patches, using a heuristic to
    ensure maximum coverage.
    If cropvalue is specified, it is treated as a flag denoting a pixel
    that has been cropped. Patch will be rejected if it has more than
    crop_fraction_allowed * prod(patchshape) pixels equal to cropvalue.
    Likewise, patches will be rejected for having more overlap_allowed
    fraction of their pixels contained in a patch already selected.
    """
    jump_cols = int(patchshape[1] * overlap_allowed)
    jump_rows = int(patchshape[0] * overlap_allowed)

    # Restrict ourselves to the rectangle containing non-cropped pixels
    if cropvalue is not None:
        rows, cols = np.where(image != cropvalue)
        rows.sort();
        cols.sort()
        active = image[rows[0]:rows[-1], cols[0]:cols[-1]]
    else:
        active = image

    rowstart = 0;
    colstart = 0

    # Array tracking where we've already taken patches.
    covered = np.zeros(active.shape, dtype=bool)
    patches = []

    while rowstart < active.shape[0] - patchshape[0]:
        # Record whether or not e've found a patch in this row, 
        # so we know whether to skip ahead.
        got_a_patch_this_row = False
        colstart = 0
        while colstart < active.shape[1] - patchshape[1]:
            # Slice tuple indexing the region of our proposed patch
            region = (slice(rowstart, rowstart + patchshape[0]),
                      slice(colstart, colstart + patchshape[1]))

            # The actual pixels in that region.
            patch = active[region]

            # The current mask value for that region.
            cover_p = covered[region]
            if cropvalue is None or \
                    frac_eq_to(patch, cropvalue) <= crop_fraction_allowed and \
                    frac_eq_to(cover_p, True) <= overlap_allowed:
                # Accept the patch.
                patches.append(patch)

                # Mask the area.
                covered[region] = True

                # Jump ahead in the x direction.
                colstart += jump_cols
                got_a_patch_this_row = True
                # print "Got a patch at %d, %d" % (rowstart, colstart)
            else:
                # Otherwise, shift window across by one pixel.
                colstart += 1

        if got_a_patch_this_row:
            # Jump ahead in the y direction.
            rowstart += jump_rows
        else:
            # Otherwise, shift the window down by one pixel.
            rowstart += 1

    # Return a 3D array of the patches with the patch index as the first
    # dimension (so that patch pixels stay contiguous in memory, in a 
    # C-ordered array).
    return np.concatenate([pat[np.newaxis, ...] for pat in patches], axis=0)


def extract_patch(classes_name):
    for patch_type in classes_name:
        if patch_type == 'test':
            input_files = recursive_glob(INPUT_DIR)
            if not exists(PREPROCESS_DIR):
                makedirs(PREPROCESS_DIR)
            else:
                shutil.rmtree(PREPROCESS_DIR)
                makedirs(PREPROCESS_DIR)
        else:
            input_files = recursive_glob(join(INPUT_DIR, patch_type))
            # shutil.rmtree(join(PREPROCESS_DIR,patch_type))
            if not exists(join(PREPROCESS_DIR, patch_type)):
                makedirs(join(PREPROCESS_DIR, patch_type))
            else:
                shutil.rmtree(join(PREPROCESS_DIR, patch_type))
                makedirs(join(PREPROCESS_DIR, patch_type))

        for f in input_files:
            # print(f)
            if (len(f.split("/")) > 2):
                class_name = f.split("/")[2]
            else:
                class_name = ""

            s = list(basename(f))

            if patch_type == 'test':
                SAVE_DIR = PREPROCESS_DIR
            else:
                SAVE_DIR = join(PREPROCESS_DIR, class_name)
            print('-----------Patch Extractor-------------')
            print(s)

            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_norm = normalize_staining(img)

            if (PATCH_TYPE == 'Random'):
                patches = patch_ex.extract_patches_2d(img_norm, (PATCH_SIZE, PATCH_SIZE), NUM_PER_PATCH)
            else:
                patches = extract_patches_overlap(img_norm, (PATCH_SIZE, PATCH_SIZE), 0.5)

            for index, patch in enumerate(patches):
                patch_name = "".join(s[:len(s) - 4])
                patch_name = patch_name + '_' + str(index)
                # if(nuclei_count(patch) >= NUCLEI_THRESHOLD):
                # patch = normalize_staining(patch)

                filename_origin = s[:len(s) - 4]

                # patch_he = patch.copy()
                for indx in range(4):
                    filename_he = list(filename_origin)
                    filename_he.append("_he" + str(index) + "_" + str(indx))
                    filename_he = "".join(filename_he)
                    patch_he = hematoxylin_eosin_aug(patch)

                    patch_he_rt = np.rot90(patch_he, np.random.randint(low=0, high=4))

                    print(join(SAVE_DIR, filename_he))
                    np.save(join(SAVE_DIR, filename_he), patch_he_rt)

                # rotate_seed = np.random.randint(1,4)
                # filename_rotate = list(filename_origin)
                # filename_rotate.append("_r"+str(index))
                # filename_rotate = "".join(filename_rotate)
                # patch_rotate = patch_he.copy()
                # patch_rotate = rotate(patch_rotate, ROTATE_DEGREE * rotate_seed)

                # filename_flip = list(filename_origin)
                # filename_flip.append("_f"+str(index))
                # filename_flip = "".join(filename_flip)
                # patch_flip = patch_he.copy()
                # patch_flip = patch_flip[:, ::-1]

                # filename_origin.append("_"+str(index))
                # filename_origin = "".join(filename_origin)

                # print(join(SAVE_DIR,filename_origin))
                # np.save(join(SAVE_DIR,filename_origin), patch)

                # print(join(SAVE_DIR,filename_flip))
                # np.save(join(SAVE_DIR,filename_flip), patch_flip)

                # print(join(SAVE_DIR,filename_rotate))
                # np.save(join(SAVE_DIR,filename_rotate), patch_rotate)
                # else:
                # print('----patch: ', patch_name , ' is not qualified')


if __name__ == "__main__":
    extract_patch(CLASSES)

    # output_files = []

    # input_files = recursive_glob(INPUT_DIR)

    # # print(input_files)

    # for f in input_files:
    #     # print(f)
    #     if(len(f.split("/")) > 2):
    #         class_name = f.split("/")[2]
    #     else:
    #         class_name = ""

    #     s = list(basename(f))
    #     SAVE_DIR = join(PREPROCESS_DIR, class_name)
    #     print('-----------Patch Extractor-------------')
    #     print(s)

    #     img = cv2.imread(f)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img_norm = normalize_staining(img)
    #     patches = patch_ex.extract_patches_2d(img_norm, (PATCH_SIZE, PATCH_SIZE), NUM_PER_PATCH)
    #     for index, patch in enumerate(patches):
    #         filename = s[:len(s)-4]
    #         filename.append("_"+str(index))
    #         filename = "".join(filename)
    #         print(join(SAVE_DIR,filename))
    #         np.save(join(SAVE_DIR,filename), patch)

    # img = cv2.imread('data/train/Benign/t0.tif')
    # img = cv2.imread('data/train/In Situ/t0.tif')
    # img = cv2.imread('data/train/Invasive/t0.tif')

    # img = cv2.imread('data/train/Normal/t0.tif')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # if SCALE != 1:
    #     img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_norm = normalize_staining(img)

    # for _ in range(AUGMENTATIONS_PER_IMAGE):
    #     img_aug = hematoxylin_eosin_aug(img_norm, low=COLOR_LO, high=COLOR_HI)
    #     # img_aug = zoom_aug(img_aug, ZOOM_VAR)

    #     # single_image_crops = get_crops_free(img_aug, PATCH_SZ, PATCHES_PER_IMAGE)
    #     single_image_crops = get_crops(img_aug, PATCH_SZ, PATCHES_PER_IMAGE)
    #     yield single_image_crops
    # print(img)

    # print('-----------')
    # patches = patch_ex.extract_patches_2d(img, (PATCH_SIZE, PATCH_SIZE), 10)
    # print(patches.shape)

    # np.save(join(SAVE_DIR,'b_t0_'), patches[0])
    # plt.imshow(patches[0], interpolation='nearest')
    # plt.show()

    # for index, patch in enumerate(patches):
    #     np.save(join(SAVE_DIR,'t0_'+str(index)), patch)
