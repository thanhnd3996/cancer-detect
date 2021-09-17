#!/usr/bin/env python3
"""Implement several encoder classes"""

from keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inception
from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg
from keras.preprocessing import image
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import GlobalAveragePooling2D, Concatenate
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Activation
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
import cv2

class ResNet:
    __name__ = "ResNet"

    def __init__(self, batch_size=32):
        self.model = ResNet50(include_top=False, weights='imagenet', pooling="avg")
        self.batch_size = batch_size
        self.data_format = K.image_data_format()

    def predict(self, x):
        if self.data_format == "channels_first":
            x = x.transpose(0, 3, 1, 2)
        x = preprocess_resnet(x.astype(K.floatx()))
        return self.model.predict(x, batch_size=self.batch_size)


class Inception:
    __name__ = "Inception"

    def __init__(self, batch_size=32):
        self.model = InceptionV3(include_top=False, weights="imagenet", pooling="avg")
        self.batch_size = batch_size
        self.data_format = K.image_data_format()

    def predict(self, x):
        if self.data_format == "channels_first":
            x = x.transpose(0, 3, 1, 2)
        x = preprocess_inception(x.astype(K.floatx()))
        return self.model.predict(x, batch_size=self.batch_size)


class VGG:
    __name__ = "VGG"

    def __init__(self, batch_size=32):
        model = VGG16(include_top=False, weights="imagenet", pooling="avg")
        x2 = GlobalAveragePooling2D()(model.get_layer("block2_conv2").output)  # 128
        x3 = GlobalAveragePooling2D()(model.get_layer("block3_conv3").output)  # 256
        x4 = GlobalAveragePooling2D()(model.get_layer("block4_conv3").output)  # 512
        x5 = GlobalAveragePooling2D()(model.get_layer("block5_conv3").output)  # 512
        x = Concatenate()([x2, x3, x4, x5])
        self.model = Model(inputs=model.input, outputs=x)
        self.batch_size = batch_size
        self.data_format = K.image_data_format()

    def predict(self, x):
        if self.data_format == "channels_first":
            x = x.transpose(0, 3, 1, 2)
        x = preprocess_vgg(x.astype(K.floatx()))
        return self.model.predict(x, batch_size=self.batch_size)

def CAE(patch_size):

    input_img = Input(shape=(patch_size, patch_size, 3))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder

def Resnet50_finetune():
    model = load_model('models/resnet50_finetune.h5')
    model.layers.pop()
    model.layers.pop()
    globalavg = GlobalAveragePooling2D()(model.layers[-1].output)
    model2 = Model(inputs=model.input, outputs=globalavg)

    return model2

def CAE_features():
    
    model = load_model('models/CAE/CAE_32_3.h5')
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    # print(model.summary())
    model.layers.pop()
    model.layers.pop()
    # print(model.summary())
    flatten = Flatten()(model.layers[-1].output)
    dense1 = Dense(512, name="dense_output1")(flatten)  # new sigmoid layer
    dense1out = Activation("relu", name="output_activation1")(dense1)
    dense2 = Dense(1, name="dense_output2")(dense1out) #new sigmoid layer
    dense2out = Activation("relu",name="output_activation2")(dense2)  # new sigmoid layer

    model2 = Model(inputs=model.input, outputs=dense2out)
    # model2 = add_new_last_layer(model.output, model.input)
    print(model2.summary())
    # print(model2.layers[8])

    for layer in model.layers[:9]:
        layer.trainable = False
    for layer in model.layers[9:]:
        layer.trainable = True

    layer_name = 'dense_output1'
    intermediate_layer_model = Model(inputs=model2.input,outputs=model2.get_layer(layer_name).output)

    # layer1 = model.get_layer("dense_output1")
    # layer2 = model.get_layer("dense_output2")
    # intermediate_output = intermediate_layer_model.predict(X_normal)
    # data_train = intermediate_output
    # intermediate_output = intermediate_layer_model.predict(X_abnormal)
    # data_test = intermediate_output

    return intermediate_layer_model

if __name__ == "__main__":
    # img_path = "data/t0.tif"
    # img = image.load_img(img_path, target_size=(300, 300))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_inception(x)

    # model = ResNet(batch_size=1)
    # preds = model.predict(x)
    # print(preds.shape)
    # print(preds)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    # print("Predicted:", decode_predictions(preds, top=3)[0])
    # Predicted: [('n02504013', 'Indian_elephant', 0.78864819), ('n01871265', 'tusker', 0.029346621), ('n02504458', 'African_elephant', 0.01768155)]

    # FILE = 'data/train/Normal/t0.tif'
    # PATCH_SIZE = 512

    # img = cv2.imread(FILE)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # patches = image.extract_patches_2d(img,(PATCH_SIZE,PATCH_SIZE), 20,1)
    

    
    model = ResNet50(include_top=False, weights='imagenet', input_shape=(512,512,3))

    flatten = Flatten()(model.layers[-1].output)
    new_layer = Dense(4, activation='softmax', name='my_dense')(flatten)
    inp = model.input
    out = new_layer

    model2 = Model(inp, out)
    model2.summary()

    print(len(model2.layers))
    print(model2.layers[174])
    # for layer in model.layers[:9]:
    #     layer.trainable = False
    # for layer in model.layers[9:]:
    #     layer.trainable = True
    
    
    


    