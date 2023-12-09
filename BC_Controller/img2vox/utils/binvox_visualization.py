# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

from mpl_toolkits.mplot3d import Axes3D


def get_volume_views(volume, save_dir, n_itr, ground_truth = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    volume = volume.squeeze().__ge__(0.5)
    volume = np.rot90(volume,1,(0,1))
    volume = np.rot90(volume,-1,(1,2))
    volume = np.rot90(volume,-2,(0,2))
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    # ax = fig.gca(projection=Axes3D.name)
    ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")

    
    save_path = os.path.join(save_dir, 'voxels-%06d.png' % n_itr)
    if ground_truth:
        save_path = os.path.join(save_dir, 'voxels-ground_truth-%06d.png' % n_itr)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return np.uint8(cv2.imread(save_path))
