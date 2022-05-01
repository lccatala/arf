#!/usr/bin/env python3
import sys
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization

from keras.utils import np_utils
from keras import optimizers
from keras import regularizers
from keras import backend as K
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

import tensorflow as tf

NORM = 'stdnorm' # 'stdnorm', 'featurescaling'
DISTRIBUTION = 7
TYPE = 'known' # 'whole', 'known', 'unknown'
FILL_STRATEGY = 'cero2mean' # 'cero2lesscontact' 'cero2mean'
IMAGE_TYPE = '3d' # '3d', 'concat-horizontal', 'concat-vertical'
FINGERS = 3
LABELS_OUT_FILE = 'labels-' + '-t' + str(DISTRIBUTION) + '-' + FILL_STRATEGY + '-' \
                + NORM + '-' + IMAGE_TYPE + '.npy'
IMAGES_OUT_FILE = 'images-' + '-t' + str(DISTRIBUTION) + '-' + FILL_STRATEGY + '-' \
                + NORM + '-' + IMAGE_TYPE + '.npy'
FINGERS = 3
TACTILE_IMAGE_ROWS = -1
TACTILE_IMAGE_COLS = -1

if DISTRIBUTION == 1:
    TACTILE_IMAGE_ROWS = 8
    TACTILE_IMAGE_COLS = 9
elif DISTRIBUTION == 2:
    TACTILE_IMAGE_ROWS = 8
    TACTILE_IMAGE_COLS = 7
elif DISTRIBUTION == 3:
    TACTILE_IMAGE_ROWS = 6
    TACTILE_IMAGE_COLS = 7
elif DISTRIBUTION == 4:
    TACTILE_IMAGE_ROWS = 4
    TACTILE_IMAGE_COLS = 7
elif DISTRIBUTION == 5:
    TACTILE_IMAGE_ROWS = 6
    TACTILE_IMAGE_COLS = 5
elif DISTRIBUTION == 6:
    TACTILE_IMAGE_ROWS = 6
    TACTILE_IMAGE_COLS = 5
elif DISTRIBUTION == 7:
    TACTILE_IMAGE_ROWS = 12
    TACTILE_IMAGE_COLS = 11



labels = np.load(LABELS_OUT_FILE)
labels_cat = np_utils.to_categorical(labels, num_classes=2)
tactile_images = np.load(IMAGES_OUT_FILE)



folds = 10
kfold = StratifiedKFold(n_splits=folds, shuffle=True)
cv_accuracies = []
cv_precision = []
cv_recall = []
cv_f1_score = []
print("# GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
for index, (train_indices, val_indices) in enumerate(kfold.split(tactile_images, labels)):
    print("Training fold " + str(index + 1) + "/" + str(folds))

    # split data
    images_train, images_val = tactile_images[train_indices], tactile_images[val_indices]
    labels_train, labels_val = labels_cat[train_indices], labels_cat[val_indices]

    # build model
    epochs = 500
    batch = 32

    learning_rate = 0.0001
    epsilon = 1e-08
    decay_rate = 0.003

    l2_reg = 0.01
    drop_prob = 0.2

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(FINGERS, TACTILE_IMAGE_ROWS, TACTILE_IMAGE_COLS),
                     data_format='channels_first',
                     use_bias=False, kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), data_format='channels_first',
                     use_bias=False, kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(1024, kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Dropout(drop_prob))

    model.add(Dense(2, activation='softmax'))

    # compile and train
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.optimizers.Adam(lr=learning_rate, epsilon=epsilon, decay=decay_rate),
                  metrics=['accuracy'])
    print('got here')
    model.fit(images_train, labels_train, epochs=epochs, batch_size=batch, verbose=1)

    # evaluate
    print("# # # Evaluating cv-fold...")
    scores = model.evaluate(images_val, labels_val, verbose=1)
    cv_accuracies.append(scores[1] * 100)

    predictions = model.predict(images_val)
    [precision, recall, f1_score, _] = precision_recall_fscore_support(np.argmax(labels_val, axis=1),
                                                                          np.argmax(predictions, axis=1),
                                                                          average='binary', pos_label=1)

    cv_precision.append(precision * 100)
    cv_recall.append(recall * 100)
    cv_f1_score.append(f1_score * 100)

    print("\naccuracy:", scores[1] * 100)
    print("precision:", precision * 100)
    print("recall:", recall * 100)
    print("f1_score:", f1_score * 100)

print("\nFinal cross-validation score:")
print(np.mean(cv_accuracies), "+/-", np.std(cv_accuracies))
print(np.mean(cv_precision), "+/-", np.std(cv_precision))
print(np.mean(cv_recall), "+/-", np.std(cv_recall))
print(np.mean(cv_f1_score), "+/-", np.std(cv_f1_score))
