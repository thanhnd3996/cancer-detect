#!/usr/bin/env python3

import os
import sys
from os.path import basename, join

import numpy as np
from keras import backend as K

from models import ResNet

# INPUT_DIR = 'data/preprocessed/train'
# INPUT_DIR = 'data/preprocessed/train_overlap'
INPUT_DIR = 'data/preprocessed/test'

# FEATURES_PATH = 'data/features/train'
# FEATURES_PATH = 'data/features/train_overlap'
FEATURES_PATH = 'data/features/test'

# CLASSES = ['Normal', 'Benign', 'Invasive', 'In Situ']
# CLASSES = ['Benign', 'Invasive', 'In Situ']
# CLASSES = ['Normal']
CLASSES = ['test']


def recursive_glob(root_dir, file_template="*.npy"):
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


def extract_feature(classes_name):
    model = ResNet()
    # model = Resnet50_finetune()
    # model = VGG()

    for class_type in classes_name:

        existed_files = recursive_glob(FEATURES_PATH)
        for f in existed_files:
            fname = basename(f)
            if fname[0] == 'n' and class_type == 'Normal':
                os.remove(f)
            elif fname[0] == 'b' and class_type == 'Benign':
                os.remove(f)
            elif fname[0] + fname[1] == 'iv' and class_type == 'Invasive':
                os.remove(f)
            elif fname[0] + fname[1] == 'is' and class_type == 'In Situ':
                os.remove(f)
            elif class_type == 'test':
                os.remove(f)

        output_files = []

        if class_type == 'test':
            input_files = recursive_glob(INPUT_DIR)
        else:
            input_files = recursive_glob(join(INPUT_DIR, class_type))
        # print(input_files)
        for f in input_files:
            # print(f)
            class_name = f.split("/")[3]
            if class_name == "Benign":
                class_name = "b"
            elif class_name == "In Situ":
                class_name = "is"
            elif class_name == "Invasive":
                class_name = "iv"
            elif class_name == "Normal":
                class_name = "n"
            else:
                class_name = "test"
            s = list(basename(f))
            if class_name == "test":
                s = [class_name] + s
            else:
                s[0] = class_name
            filename = "".join(s)
            output_files.append(join(FEATURES_PATH, filename))
        # print(output_files)
        file_list = zip(input_files, output_files)

        for j, (input_file, output_file) in enumerate(file_list):
            print('---Feature Extract---')
            # print(input_file)
            x = np.load(input_file)

            x = np.expand_dims(x, axis=0)
            print(output_file)

            # x = np.stack([x])
            # x = x.astype('float32') / 255.
            feature = model.predict(x)
            # print(x.shape)
            np.save(output_file, feature)

    K.clear_session()


if __name__ == "__main__":
    extract_feature(CLASSES)

    # output_files = []

    # input_files = recursive_glob(INPUT_DIR)
    # # print(input_files)
    # for f in input_files:
    #     # print(f)
    #     class_name = f.split("/")[3]
    #     if(class_name == "Benign"):
    #         class_name = "b"
    #     elif(class_name == "In Situ"):
    #         class_name = "is"
    #     elif(class_name == "Invasive"):
    #         class_name = "iv"
    #     elif(class_name == "Normal"):
    #         class_name = "n"
    #     else:
    #         class_name = "test"
    #     s = list(basename(f))
    #     if(class_name == "test"):
    #         s = [class_name] + s
    #     else:
    #         s[0] = class_name
    #     filename = "".join(s)
    #     output_files.append(join(FEATURES_PATH, filename))

    # file_list = zip(input_files, output_files)
    # # print(input_files)
    # # for j, inputfile in input_files:

    # model = ResNet()
    # # # model.model.summary()
    # # x = np.load('data/preprocessed/train/Normal/t0_0.npy')
    # # x = np.expand_dims(x, axis=0)
    # # feature = model.predict(x)
    # # print(feature.shape)
    # # print(feature)
    # for j, (input_file, output_file) in enumerate(file_list):
    #     print(input_file)
    #     x = np.load(input_file)
    #     x = np.expand_dims(x, axis=0)
    #     # print(output_file)
    #     feature = model.predict(x)
    #     np.save(output_file, feature)
