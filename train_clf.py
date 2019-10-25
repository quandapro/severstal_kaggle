import cv2
import numpy as np 
from efficientnet.keras import *
from classification_models.keras import Classifiers
from albumentations import *

import pandas as pd
import tensorflow as tf

import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras import backend as K
from keras import Input
from keras.models import Model
from keras.utils import *
from keras.layers import *

from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed
import matplotlib.pyplot as plt

set_random_seed(2)
np.random.seed(0)

import os
import gc
import random

IMG_SIZE = (128, 800, 3)
BATCH_SIZE = 32
train_df_path = 'severstal/mixup.csv'
data_folder = 'severstal/mixup/'
initial_lr = 0.0001
clf_model = 'efficientnetb0'

# Metrics 
def recall(y_true, y_pred, epsilon=1e-4):
    true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    return (true_pos + epsilon) / (all_pos + epsilon)

def precision(y_true, y_pred, epsilon=1e-4):
    true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pred_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return (true_pos + epsilon) / (pred_pos + epsilon)

def F1(y_true, y_pred, epsilon=1e-4):
    precision_m = precision(y_true, y_pred)
    recall_m = recall(y_true, y_pred)
    return 2*((precision_m*recall_m)/(precision_m + recall_m + epsilon))

df = pd.read_csv(train_df_path)
df['ImageId'] = df['ImageId_ClassId'].map(lambda x : x.split('_')[0])
df['ClassId'] = df['ImageId_ClassId'].map(lambda x : x.split('_')[1])
df = df.drop(['ImageId_ClassId'], axis = 1)
df['has_label'] = df['EncodedPixels'].map(lambda x : 1 if isinstance(x,str) else 0)
df.head(5)

i = 0
image_id = []
d = {}

while i < len(df['ImageId']):
    img_id = df['ImageId'][i]
    image_id.append(img_id)
    tmp = np.array([df['has_label'][i], df['has_label'][i + 1], df['has_label'][i + 2], df['has_label'][i + 3]])
    d[img_id] = tmp
    i += 4

# Datagen
def load_target_image(path, grayscale = False, color_mode = 'rgb', target_size = (IMG_SIZE[0], IMG_SIZE[1], 3),
                     interpolation = 'nearest'):
    
    return load_img(path=path, grayscale=grayscale, color_mode=color_mode,
                   target_size=target_size, interpolation=interpolation)

class Datagen(Sequence):
    
    def __init__(self, dataframe, fns, target_dir, batch_size, target_size, use_aug=False, preprocessing_function=None, p=0.8):
        self.dataframe = dataframe
        self.fns = fns
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.use_aug = use_aug
        # self.preprocessing_function = preprocessing_function or (lambda x: x)
        self.aug = \
            Compose([
                    HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5),
                    ShiftScaleRotate(shift_limit=0.1625, scale_limit=0., rotate_limit=0, p=1.),
                    OneOf([
                        GridDistortion(p=0.5, num_steps=5, distort_limit=[-0.3, 0.3]),
                        ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03), 
                        OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
                    ], p=0.3),
                    OneOf([
                        RandomBrightness(limit=0.2, p=0.5),
                        RandomContrast(limit=0.2, p=0.5),
                        RandomGamma(gamma_limit=(80,120), p=0.5),
                        CLAHE(p=0.65, tile_grid_size=(11, 11))
                    ], p=0.3)], p=p)

    def __len__(self):
        return int(np.ceil(len(self.fns) / float(self.batch_size)))

    def augment(self, image):
        augmented = self.aug(image=image)
        image_heavy = augmented['image']
        return image_heavy

    def __getitem__(self, idx):
        current_fns = self.fns[idx*self.batch_size:(idx + 1)*self.batch_size]
        batch_image = np.empty((self.batch_size, self.target_size[0], self.target_size[1], 3), dtype='float32')
        batch_labels = np.empty((self.batch_size, 4), dtype='float32')           

        for i in range(len(current_fns)):
            fn = current_fns[i]
            cur_img = np.asarray(load_target_image(path=os.path.join(self.target_dir, fn)))
            label = self.dataframe[fn]
            if self.use_aug:
                cur_img = self.augment(cur_img)

            # cur_img = self.preprocessing_function(cur_img)
            batch_image[i] = cur_img
            batch_labels[i] = label

        return batch_image / 255., batch_labels

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=2019)

X = np.array(image_id)
fold = 0

import math
def scheduler(epoch, lr):
    if lr >= 5e-6:
        return initial_lr*math.exp(-epoch*0.05)
    return lr

for train_index, test_index in kf.split(X):

    base_clf = EfficientNetB0(input_shape=IMG_SIZE, weights='imagenet', include_top=False)
    x = base_clf.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4, activation='sigmoid', kernel_initializer='he_normal')(x)
    clf = Model(inputs=base_clf.input, outputs=x)
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, verbose=1, monitor='val_loss', mode='min'),
        keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
        keras.callbacks.ModelCheckpoint('models/defect_clf/{}_clf_fold_{}_v2.h5'.format(clf_model, fold), 
                                        verbose=1, 
                                        save_best_only=True, 
                                        save_weights_only=True,
                                        monitor='val_loss',
                                        mode='min')
    ]
    
    train_x = X[train_index]
    valid_x = X[test_index]
    
    train_aug_gen = Datagen(d, train_x, data_folder, BATCH_SIZE, IMG_SIZE, use_aug=True, preprocessing_function=None, p=1.)
    valid_aug_gen = Datagen(d, valid_x, data_folder, BATCH_SIZE, IMG_SIZE, use_aug=False, preprocessing_function=None)
        
    clf.compile(optimizer=keras.optimizers.Adam(lr=initial_lr), loss='binary_crossentropy', metrics=['acc', 
                                                                                                      precision, 
                                                                                                      recall, 
                                                                                                      F1])
    
    clf.fit_generator(train_aug_gen, 
                        steps_per_epoch=len(train_aug_gen),
                        epochs=125,
                        validation_data=valid_aug_gen,
                        validation_steps=len(valid_aug_gen),
                        callbacks=callbacks,
                        workers=4)

    fold += 1