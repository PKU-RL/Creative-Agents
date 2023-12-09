# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import json
import numpy as np
import os
import random
import scipy.io
import scipy.ndimage
import sys
import torch.utils.data.dataset

from datetime import datetime as dt
from enum import Enum, unique

import utils.binvox_rw

from PIL import Image

@unique
class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


VoxelAugmented_LEN = 960100
VoxelAugmented_TRAIN_LEN = 850100

TEST_RATIO = 15

def process_vox_id(ori_vox):
    return np.where(ori_vox>0,1,0)

class VoxelAugmented_RGB_Data(torch.utils.data.dataset.Dataset):

    def __init__(self, vox_path, img_path, train = True , transforms=None, pipline = False, pip_img_pth = None, pip_vox_pth = None):
        # print(vox_path)
        self.vox_path = vox_path
        self.img_path = img_path
        self.transforms = transforms 
        self.pipline = pipline
        self.pip_img_pth = pip_img_pth
        self.pip_vox_pth = pip_vox_pth
        self.test_seg = 400
        
        # for test training set
        self.train_mode = True
        self.train_mode = train

        # self.v_file_names = os.listdir(vox_path)
        # self.v_file_names.sort()
        # # print(self.v_file_names)
        # self.i_file_names = os.listdir(img_path)
        # self.i_file_names.sort()

        # assert(len(self.v_file_names) == len(self.i_file_names))
        # self.total_len = len(self.v_file_names)

    def __len__(self):
        # return len(self.file_list)
        if self.pipline:
            test_pth = self.pip_img_pth
            test_pth_fils = os.listdir(test_pth)
            png_files = [file for file in test_pth_fils if file.endswith(".png")]
            return len(png_files)
        if self.train_mode:
            return VoxelAugmented_TRAIN_LEN
        else:
            return int((VoxelAugmented_LEN - VoxelAugmented_TRAIN_LEN) / self.test_seg)-1


    def __getitem__(self, idx):
        info, rendering_images, volume, bounding_box = self.get_datum(idx)
        if self.transforms:
            rendering_images = self.transforms(rendering_images, bounding_box)
        if self.train_mode:
            return rendering_images, volume
        else:
            return rendering_images, volume, info
    

    def match_color_aug_dataset_jpg(self,quary_i, ord_path = "./img2vox/fixed_dataset_ord.npy", img_dir = "./datasets/image_augmented", vox_dir = "./datasets/voxel_augmented", item2rgb_path = "./img2vox/ord2rgb.npy"):
        v_ord_list = np.load(ord_path)
        v_ord  = v_ord_list[quary_i][0]-1
        img_ord = v_ord_list[quary_i][1]
        infile_ord = v_ord_list[quary_i][2]
        # print(v_ord_list.shape)

        vox_path = os.path.join(vox_dir,"CraftAssist_V_auged_{}.npy".format(v_ord))
        vox = np.load(vox_path)
        img_path = os.path.join(img_dir,"{}_{}.jpg".format(img_ord,infile_ord))
        # img = np.load(img_path)
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = np.array(img)

        rotate_vox = vox[int(infile_ord/4)]
        for _ in range(infile_ord%4):
            rotate_vox = np.rot90(rotate_vox,-1,(1,2))
        
        ord2rgb = np.load(item2rgb_path)
        rotate_vox = rotate_vox.astype(np.int32)
        colored_vox = ord2rgb[rotate_vox] / 255
        rotate_vox = (np.where(rotate_vox>0,1,0)).reshape(32,32,32,1)
        colored_vox = np.concatenate((colored_vox,rotate_vox),axis = 3)

        return  img/255, colored_vox

    def get_datum(self, idx):
        rel_idx = idx
        if not self.train_mode and not self.pipline:
            rel_idx = idx*self.test_seg + VoxelAugmented_TRAIN_LEN
        
        if not self.pipline:
            rendering_image, volume = self.match_color_aug_dataset_jpg(rel_idx,img_dir = self.img_path, vox_dir=self.vox_path)
        info = []
        if self.pipline:
            test_pth = self.pip_img_pth
            test_pth_fils = os.listdir(test_pth)
            png_files = [file for file in test_pth_fils if file.endswith(".png")]
            
            if not self.train_mode:
                im = Image.open(os.path.join(test_pth,png_files[idx]))
                image=im.resize((512,512))
                rendering_image = (np.array(image) /255.)[:,:,0:3]
                # print(rendering_image.shape)
                volume = np.zeros((32,32,32,4))
                info.append(png_files[idx].split('.')[0])

        return info, np.asarray([rendering_image]), volume, None
    


DATASET_LOADER_MAPPING = {
    "MC_RGB" :VoxelAugmented_RGB_Data
}  # yapf: disable
