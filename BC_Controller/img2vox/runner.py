#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import logging
import matplotlib
import multiprocessing as mp
import numpy as np
import os
import sys
# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

from argparse import ArgumentParser
from datetime import datetime as dt
from pprint import pprint

from config import cfg
from core.train import train_net
from core.test_mc import test_net_mc_RGB


OUPUT_DIR = "./img2vox/test_RGB"

def get_args_from_command_line(ckpt_path = None):
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        help='GPU device id to use [cuda0]',
                        default=cfg.CONST.DEVICE,
                        type=str)

    parser.add_argument('--rand', dest='randomize', help='Randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--pipline', dest='pipline', help='use img2vox model in our pipline', action='store_true')
    parser.add_argument('--gen_gif', dest='gen_gif', help='MC generate gif', action='store_true')
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='name of the net',
                        default=cfg.CONST.BATCH_SIZE,
                        type=int)
    parser.add_argument('--epoch', dest='epoch', help='number of epoches', default=cfg.TRAIN.NUM_EPOCHES, type=int)
    
    # resume model path
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=ckpt_path)
    # if you are using 512*512 input and 512*512 model, modify the __C.CONST.IMG_W and __C.CONST.IMG_H in config.py
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    args = parser.parse_args()
    return args


def run_img2vox(pipline = False, resume = True):
    if resume:
        if cfg.CONST.IMG_W ==224:
            assert (cfg.CONST.IMG_H ==224)
            ckpt_path = "./models/img2vox-224.pth"
        else:
            assert (cfg.CONST.IMG_H ==512)
            ckpt_path = "./models/img2vox-512.pth"
        # Get args from command line
        args = get_args_from_command_line(ckpt_path)
    else:
        args = get_args_from_command_line()
    
    if not pipline:
        pipline = args.pipline
    if pipline:
        args.test = True

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if not args.randomize:
        np.random.seed(cfg.CONST.RNG_SEED)
    if args.batch_size is not None:
        cfg.CONST.BATCH_SIZE = args.batch_size
    if args.epoch is not None:
        cfg.TRAIN.NUM_EPOCHES = args.epoch
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
        if not args.test:
            cfg.TRAIN.RESUME_TRAIN = True

    # Print config
    print('Use config:')
    pprint(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    # Start train/test process
    if not args.test:
        train_net(cfg)
    else:
        if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
            cfg.DATASET.TRAIN_DATASET                   = 'MC_RGB'
            cfg.DATASET.TEST_DATASET                  = 'MC_RGB'
            test_net_mc_RGB(cfg,args.gen_gif,output_dir=OUPUT_DIR,pipline=pipline)
        else:
            print('[FATAL] %s Please specify the file path of checkpoint.' % (dt.now()))
            sys.exit(2)


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on 'https://github.com/hzxie/Pix2Vox'")

    # Setup logger
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)

    run_img2vox()
