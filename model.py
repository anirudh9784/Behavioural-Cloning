import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from utils import INPUT_SHAPE, batch_generator
import argparse
import os
import csv
import cv2
np.random.seed(0)


if __name__ == '__main__':
    images = []
    angles = []
    '''with open('./data' + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            center_image = cv2.imread(line[0].split('/')[-1])
            center_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
            images.append(center_image_rgb)
            angles.append(float(line[3]))
            images.append(cv2.flip(center_image_rgb, 1))
            angles.append(-float(line[3]))
            left = cv2.imread(line[1].split('/')[-1])
            left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
            images.append(left_rgb)
            angles.append(float(line[3])+0.25)
            images.append(cv2.flip(left_rgb, 1))
            angles.append(-float(line[3])+0.25)
            right = cv2.imread(line[2].split('/')[-1])
            right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
            images.append(right_rgb)
            angles.append(float(line[3])-0.25)
            images.append(cv2.flip(right_rgb, 1))
            angles.append(-float(line[3])-0.25)
    print(len(angles))
    '''
    with open('driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            center_image = cv2.imread(line[0].split('/')[-1])
            center_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
            images.append(center_image_rgb)
            angles.append(float(line[3]))
            images.append(cv2.flip(center_image_rgb, 1))
            angles.append(-float(line[3]))
            left = cv2.imread(line[1].split('/')[-1])
            left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
            images.append(left_rgb)
            angles.append(float(line[3])+0.25)
            images.append(cv2.flip(left_rgb, 1))
            angles.append(-float(line[3])+0.25)
            right = cv2.imread(line[2].split('/')[-1])
            right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
            images.append(right_rgb)
            angles.append(float(line[3])-0.25)
            images.append(cv2.flip(right_rgb, 1))
            angles.append(-float(line[3])-0.25)

    X = images
    y = angles
    print(len(y))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.summary()

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data = (X_val, y_val), nb_epoch=10, shuffle=True) #, shuffle=True, validation_split=0.1

    model.save('model.h5')