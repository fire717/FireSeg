 
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
import cv2
import albumentations as A
import json
import platform


def fixed_position_aug(img, img1=False):

    # img = A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.1, 
    #                                        contrast_limit=0.1, p=0.5), 
    #                 A.HueSaturationValue(hue_shift_limit=10, 
    #                     sat_shift_limit=10, val_shift_limit=10,  p=0.5)], 
    #                 p=0.5)(image=img)['image']



    img = A.RGBShift(r_shift_limit=10,
                        g_shift_limit=10,
                        b_shift_limit=10,
                        p=0.5)(image=img)['image']

    img = A.OneOf([
                    # A.GaussianBlur(blur_limit=3, p=0.1),
                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.4)], 
                        p=0.5)(image=img)['image']

    # img = A.GridDropout(ratio=0.1+random.random()*0.2, 
    #             p=0.3)(image=img)['image']

    # if img1:
    #     img = A.RGBShift(r_shift_limit=100,
    #                         g_shift_limit=100,
    #                         b_shift_limit=100,
    #                         p=0.3)(image=img)['image']

    # if random.random()<0.3:
    #     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


    return img


def change_position_aug(img1,mask):

    transform = A.Compose([
            A.ShiftScaleRotate(
                                shift_limit=0.1,
                                scale_limit=0.1,
                                rotate_limit=20,
                                interpolation=cv2.INTER_LINEAR,
                                border_mode=cv2.BORDER_CONSTANT,
                                 value=0, mask_value=0,
                                p=0.6),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
        ])


    transformed = transform(image=img1, mask=mask)
    img1 = transformed['image']
    mask = transformed['mask']
    return img1,mask


###### 1.Data aug
class TrainDataAug:
    def __init__(self, img_size):
        self.h = img_size[0]
        self.w = img_size[1]

    def __call__(self, img,label):

        # if random.random()<0.5:
        #     img = fixed_position_aug(img)


        # if random.random()<0.5:
        #     img,label = change_position_aug(img,label)
        
        return img,label


class TestDataAug:
    def __init__(self, img_size):
        self.h = img_size[0]
        self.w = img_size[1]

    def __call__(self, img,label):
        # opencv img, BGR
        # new_width, new_height = self.size[0], self.size[1]
        
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


        # img = A.Resize(self.h,self.w,cv2.INTER_LANCZOS4,p=1)(image=img)['image']
        # img = Image.fromarray(img)
        return img,label



