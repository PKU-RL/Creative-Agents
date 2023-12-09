# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict

__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
#
__C.DATASETS                                = edict()
# __C.DATASETS.BC_VOXEL_PATH                  = "./datasets/voxel_augmented"
# __C.DATASETS.BC_IMG_PATH                    = "./datasets/image_augmented"

#HERE
__C.DATASETS.BC_VOXEL_PATH                  = "/home/ps/Desktop/DDPM/MC_image/voxel_augmented"
__C.DATASETS.BC_IMG_PATH                    = "/home/ps/Desktop/Pix2Vox/jpg_file"

# __C.DATASETS.PIP_VOXEL_PATH                  = "./image_augmented"
__C.DATASETS.PIP_IMG_PATH                    = "./results/image"

#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET                   = 'MC_RGB'
__C.DATASET.TEST_DATASET                    = 'MC_RGB'


#
# Common
#


__C.CONST                                   = edict()
__C.CONST.PIP_VOX_PATH                      = "./results/voxel"
__C.CONST.DEVICE                            = '0'
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 224       # Image width for input
__C.CONST.IMG_H                             = 224       # Image height for input

# set 512 to use/train 512 input model
# __C.CONST.IMG_W                             = 512
# __C.CONST.IMG_H                             = 512 


__C.CONST.N_VOX                             = 32
__C.CONST.BATCH_SIZE                        = 3
__C.CONST.N_VIEWS_RENDERING                 = 1         
__C.CONST.CROP_IMG_W                        = 512       
__C.CONST.CROP_IMG_H                        = 512       

#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = './img2vox/output'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False
__C.NETWORK.USE_REFINER                     = False
__C.NETWORK.USE_MERGER                      = False

#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = True
__C.TRAIN.NUM_WORKER                        = 4             # number of data workers
__C.TRAIN.NUM_EPOCHES                       = 1000
__C.TRAIN.BRIGHTNESS                        = .4
__C.TRAIN.CONTRAST                          = .4
__C.TRAIN.SATURATION                        = .4
__C.TRAIN.NOISE_STD                         = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                            = 'adam'        # available options: sgd, adam
__C.TRAIN.EPOCH_START_USE_REFINER           = 0
__C.TRAIN.EPOCH_START_USE_MERGER            = 0
__C.TRAIN.ENCODER_LEARNING_RATE             = 1e-3
__C.TRAIN.DECODER_LEARNING_RATE             = 1e-3
__C.TRAIN.REFINER_LEARNING_RATE             = 1e-3
__C.TRAIN.MERGER_LEARNING_RATE              = 1e-4
__C.TRAIN.ENCODER_LR_MILESTONES             = [150]
__C.TRAIN.DECODER_LR_MILESTONES             = [150]
__C.TRAIN.REFINER_LR_MILESTONES             = [150]
__C.TRAIN.MERGER_LR_MILESTONES              = [150]
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .5
__C.TRAIN.SAVE_FREQ                         = 5            # weights will be overwritten every save_freq epoch
__C.TRAIN.UPDATE_N_VIEWS_RENDERING          = False

#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                       = [.2, .3, .4, .5]
