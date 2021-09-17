#!/usr/bin/env python3
import keras
import numpy as np
from keras import backend as K
from keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet
from keras.layers import Dense, Flatten
from keras.models import Model

from feature_extractor import recursive_glob

INPUT_DIR = 'data/retrain'
# INPUT2_DIR = 'data/preprocessed/train'
# FEATURE_PATH = 'data/features/train'
FILES_PER_CLASS = 8

if __name__ == "__main__":
    # [syn_train,syn_train_label,syn_test,syn_test_label] = prepare_synthetic_data()
    # print ("[INFO] SYNTHETIC DATA : (Normal, Anomalies) dimensions: ",syn_train.shape,syn_test.shape)
    # print(syn_train.shape, syn_test.shape)
    # print(syn_train_label.shape, syn_test_label.shape)
    # print(syn_test, syn_test_label)

    input_files = recursive_glob(INPUT_DIR)
    # input_files = recursive_glob(INPUT2_DIR)
    # input_files = recursive_glob(FEATURE_PATH)
    # print(input_files)

    X = []
    y = []

    n_files = []
    b_files = []
    is_files = []
    iv_files = []
    files = []

    # for f in input_files:
    #     class_name = f.split("/")[2]
    #     if(class_name == 'Normal'):
    #         y.append(0)
    #     elif(class_name == 'Benign'):
    #         y.append(1)
    #     elif(class_name == 'In Situ'):
    #         y.append(2)
    #     elif(class_name == 'Invasive'):
    #         y.append(3)
    #     x = np.load(f)
    #     X.append(x)
    #     del x

    # X = np.stack(X)
    # y = np.stack(y)

    # X = preprocess_resnet(X.astype(K.floatx()))
    # y = keras.utils.to_categorical(y, num_classes=4)

    # print(X.shape)
    # print(y.shape)

    def loadBatchData(files, class_label):
        X = []
        y = []
        for f in files:
            x = np.load(f)
            y.append(class_label)
            X.append(x)
        X = np.stack(X)
        y = np.stack(y)
        return X, y


    for f in input_files:
        class_name = f.split("/")[2]
        if class_name == 'Normal':
            n_files.append(f)
        elif class_name == 'Benign':
            b_files.append(f)
        elif class_name == 'In Situ':
            is_files.append(f)
        elif class_name == 'Invasive':
            iv_files.append(f)

    num_epoch = 10
    num_iters = len(n_files) // FILES_PER_CLASS
    print(num_iters)

    n_files = n_files[:len(n_files) // FILES_PER_CLASS * FILES_PER_CLASS]
    b_files = b_files[:len(b_files) // FILES_PER_CLASS * FILES_PER_CLASS]
    is_files = is_files[:len(is_files) // FILES_PER_CLASS * FILES_PER_CLASS]
    iv_files = iv_files[:len(iv_files) // FILES_PER_CLASS * FILES_PER_CLASS]

    # print(np.reshape(np.array(n_files),(-1,FILES_PER_CLASS)))
    n_files = np.reshape(np.array(n_files), (-1, FILES_PER_CLASS))
    b_files = np.reshape(np.array(b_files), (-1, FILES_PER_CLASS))
    is_files = np.reshape(np.array(is_files), (-1, FILES_PER_CLASS))
    iv_files = np.reshape(np.array(iv_files), (-1, FILES_PER_CLASS))

    # print(X.shape)

    model = ResNet50(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
    # model = ResNet50(include_top=False, input_shape=(512,512,3))

    flatten = Flatten()(model.layers[-1].output)
    new_layer = Dense(4, activation='softmax', name='my_dense')(flatten)
    inp = model.input
    out = new_layer

    model2 = Model(inp, out)
    # model2.summary()

    # print(model2.layers[174])
    for layer in model2.layers[:174]:
        layer.trainable = False
    for layer in model2.layers[174:]:
        layer.trainable = True

    Adam_opt = keras.optimizers.Adam(lr=0.00001)
    # SGD_optimizer = keras.optimizers.SGD(lr=0.001)
    model2.compile(optimizer=Adam_opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # model2.fit(X, y, nb_epoch=1, verbose=1, shuffle=True, validation_split=0.2, batch_size=32)
    for e in range(num_epoch):
        print("epoch %d" % e)
        val_iters = num_iters - 2
        val_x = []
        val_y = []
        for step in range(num_iters):
            # print("step: %d" %step)
            n_x, n_y = loadBatchData(n_files[step], 0)
            b_x, b_y = loadBatchData(b_files[step], 1)
            is_x, is_y = loadBatchData(is_files[step], 2)
            iv_x, iv_y = loadBatchData(iv_files[step], 3)
            train_x = np.concatenate((n_x, b_x, is_x, iv_x))
            train_y = np.concatenate((n_y, b_y, is_y, iv_y))

            train_x = preprocess_resnet(train_x.astype(K.floatx()))
            train_y = keras.utils.to_categorical(train_y, num_classes=4)
            # print(train_y)
            if step <= val_iters:
                model2.fit(train_x, train_y, nb_epoch=1, verbose=2, shuffle=True)
            else:
                val_x.append(train_x)
                val_y.append(train_y)
        val_x = np.stack(val_x[0])
        val_y = np.stack(val_y[0])
        validation = model2.evaluate(val_x, val_y)
        print(validation)

    model2.save('models/resnet50_finetune.h5')

    # y = np.stack(y)
    # X = np.stack(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # # print(X_train.shape)
    # # print(X_test.shape)

    # model = CAE(patch_size=FILES_PER_CLASS)
    # Adam_opt = keras.optimizers.Adam(lr=0.0005)

    # model.compile(optimizer=Adam_opt, loss='MSE', metrics=['accuracy'])
    # print(model.summary())

    # X_train = X_train.astype('floatFILES_PER_CLASS') / 255.
    # X_train = np.reshape(X_train, (len(X_train), FILES_PER_CLASS, FILES_PER_CLASS, 3))
    # X_test = X_test.astype('floatFILES_PER_CLASS') / 255.
    # X_test = np.reshape(X_test, (len(X_test), FILES_PER_CLASS, FILES_PER_CLASS, 3))

    # model.fit(X_train, X_train,
    #                 epochs=500,
    #                 FILES_PER_CLASS=FILES_PER_CLASS,
    #                 shuffle=True,
    #                 validation_data=(X_test, X_test),
    #                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder3')])

    # model.save('models/CAE/CAE_FILES_PER_CLASS_3.h5')

    K.clear_session()
