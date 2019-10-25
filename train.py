import segmentation_models as sm
import cv2
import numpy as np 

import pandas as pd
import tensorflow as tf

import keras
from keras.optimizers import *
from keras import backend as K
from keras import Input
from keras.models import Model
from keras.callbacks import *
from keras.metrics import *
from keras_gradient_accumulation import GradientAccumulation

from tensorflow import set_random_seed
import matplotlib.pyplot as plt

set_random_seed(2)
np.random.seed(0)

import os
import gc
import random

from utils import *
from config import *
from generator import *
from accumulate import *

import math
from sklearn.model_selection import KFold

def scheduler(epoch, lr):
    if lr >= 5e-6:
        return initial_lr*math.exp(-epoch*0.05)
    return lr

def main():
    df = pd.read_csv(train_df_path)

    df['ImageId'] = df['ImageId_ClassId'].map(lambda x : x.split('_')[0])
    df['ClassId'] = df['ImageId_ClassId'].map(lambda x : x.split('_')[1])
    df = df.drop(['ImageId_ClassId'], axis = 1)

    df['has_label'] = df['EncodedPixels'].map(lambda x : 1 if isinstance(x,str) else 0)

    train_fns = np.unique(df['ImageId'])

    kf = KFold(n_splits=5, shuffle=True, random_state=2019)

    X = np.array(train_fns)
    Fold = 0

    val_dice_coef = np.zeros((5))

    for train_index, test_index in kf.split(X):
        gc.collect()
        K.clear_session()
        model = sm.Unet(model_name, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), classes=5, activation='softmax')
        print('************************Fold: {}************************'.format(Fold))
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_Dice_Coef', patience=10, mode='max', min_delta=0.02),
            keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
            keras.callbacks.ModelCheckpoint('./{}/unet_{}_fold_{}_mixup_full_size.h5'.format(model_name, model_name, Fold), 
                                            verbose=1, 
                                            save_best_only=True, 
                                            save_weights_only=True,
                                            monitor='val_Dice_Coef',
                                            mode='max')
        ]

        train_x = X[train_index]
        valid_x = X[test_index]

        train_aug_gen = Datagen(df, train_x, data_path, BATCH_SIZE, IMG_SIZE, use_aug=True, preprocessing_function=None, p=1.)
        valid_aug_gen = Datagen(df, valid_x, data_path, BATCH_SIZE, IMG_SIZE, use_aug=False, preprocessing_function=None)

        ops = Adam(lr=initial_lr)
        model.compile(optimizer=ops, loss='categorical_crossentropy', metrics=[Dice_Coef,
                                                                                keras.metrics.CategoricalAccuracy()])
                                                        
        if os.path.isfile('./{}/unet_{}_fold_{}_mixup_full_size.h5'.format(model_name, model_name, Fold - 1)):
            model.load_weights('./{}/unet_{}_fold_{}_mixup_full_size.h5'.format(model_name, model_name, Fold - 1))

        hist = model.fit_generator(train_aug_gen, 
                            steps_per_epoch=len(train_aug_gen),
                            epochs=100,
                            validation_data=valid_aug_gen,
                            validation_steps=len(valid_aug_gen),
                            callbacks=callbacks,
                            use_multiprocessing=False,
                            workers=4,
                            max_queue_size=4)
        val_dice_coef[Fold] = np.max(hist.history['val_Dice_Coef'])
        Fold += 1

    print(val_dice_coef)
    print(val_dice_coef.mean())


if __name__ == '__main__':
    main()