import random
from keras.preprocessing.image import ImageDataGenerator, load_img
from multiprocessing import Pool
import os
from config import *
from utils import *
import cv2

from albumentations import *

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
                    ], p=0.3)], 
            additional_targets={'mask0': 'mask', 'mask1': 'mask', 'mask2': 'mask', 'mask3': 'mask', 'mask4': 'mask'},
            p=p)

    def load_target_image(self, path, grayscale = False, color_mode = 'rgb', target_size = (256, 1600, 3),
                          interpolation = 'nearest'):
        return load_img(path = path, grayscale = grayscale, color_mode = color_mode,
                        target_size = target_size, interpolation = interpolation)

    def __len__(self):
        return int(np.ceil(len(self.fns) / float(self.batch_size)))

    def augment(self, image, masks):
        mask_heavy = np.empty((self.target_size[0], self.target_size[1], 5))
        augmented = self.aug(image=image, mask0=masks[:,:,0], mask1=masks[:,:,1], mask2=masks[:,:,2], mask3=masks[:,:,3], mask4=masks[:,:,4])
        image_heavy = augmented['image']
        mask_heavy[:,:,0] = augmented['mask0']
        mask_heavy[:,:,1] = augmented['mask1']
        mask_heavy[:,:,2] = augmented['mask2']
        mask_heavy[:,:,3] = augmented['mask3']
        mask_heavy[:,:,4] = augmented['mask4']
        return image_heavy, mask_heavy

    def __getitem__(self, idx):
        current_fns = self.fns[idx*self.batch_size:(idx + 1)*self.batch_size]
        batch_image = np.empty((self.batch_size, self.target_size[0], self.target_size[1], 3), dtype='float32')
        batch_masks = np.empty((self.batch_size, self.target_size[0], self.target_size[1], 5), dtype='float32')           

        for i in range(len(current_fns)):
            fn = current_fns[i]
            cur_img = np.asarray(self.load_target_image(path = self.target_dir + fn))
            masks = np.zeros((self.target_size[0], self.target_size[1], 5), dtype='uint8')
            masks = merge_masks(fn, self.dataframe.copy(), reshape = self.target_size)
            gc.collect()
            if self.use_aug:
                cur_img, masks = self.augment(cur_img, masks)
            batch_image[i] = cur_img
            batch_masks[i] = masks

        return batch_image / 255., batch_masks