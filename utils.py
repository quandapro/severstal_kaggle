import segmentation_models as sm
import cv2
import numpy as np 

import pandas as pd
from tqdm import tqdm_notebook
import tensorflow as tf

import keras
from keras.optimizers import Optimizer
from keras.legacy import interfaces
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

from config import *

def rle_encoding(mask):
    
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels,[0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if len(runs) % 2:
        runs = np.append(runs,len(pixels))
    runs[1::2] -= runs[0::2]
    
    return ' '.join(str(x) for x in runs)

def rle_decoding(rle, mask_shape = (256,1600)):
    strs = rle.split(' ')
    starts = np.asarray(strs[0::2], dtype = int) - 1
    lengths = np.asarray(strs[1::2], dtype = int)
    ends = starts + lengths
    
    mask = np.zeros(mask_shape[0] * mask_shape[1], dtype = np.uint8)
    for s,e in zip(starts, ends):
        mask[s:e] = 1
    return mask.reshape(mask_shape, order = 'F')

def merge_masks(image_id, df, mask_shape = (256,1600), reshape = None):
    
    rles = df[df['ImageId'] == image_id].EncodedPixels.iloc[:]
    depth = 4
    if reshape:
        masks = np.zeros((*reshape, depth + 1), dtype = np.uint8)
    else:
        masks = np.zeros((mask_shape[0], mask_shape[1],depth + 1), dtype = np.uint8)
    
    for idx in range(depth):
        if isinstance(rles.iloc[idx], str):
            if reshape:
                cur_mask = rle_decoding(rles.iloc[idx], mask_shape)
                cur_mask = cv2.resize(cur_mask, (reshape[1], reshape[0]))
                masks[:,:,idx + 1] += cur_mask
            else:         
                masks[:,:,idx + 1] += rle_decoding(rles.iloc[idx], mask_shape)
    masks[:,:,0] = np.array((masks[:,:,1] + masks[:,:,2] + masks[:,:,3] + masks[:,:,4]) == 0) 
    return masks   

# def merge_masks(image_id, df, mask_shape = (256,1600), reshape = None):
    
#     rles = df[df['ImageId'] == image_id].EncodedPixels.iloc[:]
#     depth = rles.shape[0]
#     if reshape:
#         masks = np.zeros((*reshape, depth), dtype = np.uint8)
#     else:
#         masks = np.zeros((mask_shape[0], mask_shape[1],depth), dtype = np.uint8)
    
#     for idx in range(depth):
#         if isinstance(rles.iloc[idx], str):
#             if reshape:
#                 cur_mask = rle_decoding(rles.iloc[idx], mask_shape)
#                 cur_mask = cv2.resize(cur_mask, (reshape[1], reshape[0]))
#                 masks[:,:,idx] += cur_mask
#             else:         
#                 masks[:,:,idx] += rle_decoding(rles.iloc[idx], mask_shape)
#     return masks  

def Dice_Coef(y_true, y_pred, epsilon=1e-4):
    
    y_true_f = K.flatten(y_true[:,:,:,1:4])
    y_pred_f = K.flatten(y_pred[:,:,:,1:4])

    intersection = K.sum(y_true_f * y_pred_f)

    return 2*intersection / (K.sum(y_true_f) + K.sum(y_pred_f) + epsilon)

def Dice_Loss(y_true, y_pred):
    return 1 - Dice_Coef(y_true, y_pred)

def soft_dice_Loss(y_true, y_pred, epsilon=1e-4):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2*intersection / (K.sum(y_true_f) + K.sum(y_pred_f) + epsilon))

def bce_dice_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) + Dice_Loss(y_true, y_pred)

def categorical_dice_loss(y_true, y_pred):
    return keras.losses.categorical_crossentropy(y_true, y_pred) + soft_dice_Loss(y_true, y_pred)

