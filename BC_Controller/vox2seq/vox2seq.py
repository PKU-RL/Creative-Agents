import torch
import torch.nn as nn

import pickle
from model import ResNet3D, BasicBlock
import numpy as np
import os
import skimage.measure as measure

H = 32
W = 32
L = 32
MIN_SEG_SIZE = 5

# DXYZ = [[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[0,1,1],[0,1,-1],[0,-1,-1],[0,-1,1]]
DXYZ = [[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
DXYZ = np.array(DXYZ)


non_place_items=[
    6,
    8,
    9,
    10,
    11,
    26,
    27,
    28,
    31,
    32,
    34,
    37,
    38,
    39,
    40,
    50,
    51,
    55,
    59,
    63,
    64,
    65,
    66,
    68,
    69,
    71,
    75,
    76,
    77,
    81,
    83,
    92,
    93,
    94,
    104,
    105,
    106,
    111,
    115,
    119,
    122,
    132,
    141,
    142,
    143,
    157,
    166,
    171,
    175,
    178,
    193,
    194,
    195,
    196,
    197,
    198,
    217,
]


def convert_vox(vox_path,save_dir):
    vox = np.load(vox_path)
    vox = np.transpose(vox,(0,2,1))
    vox = np.where(vox>0,1,0)
    np.save(os.path.join(save_dir,os.path.basename(vox_path)),vox)


def check_bound(xyz,input_shape):
    for i in range(3):
        # print(xyz[i])
        # print(input_shape)
        if xyz[i]<0 or xyz[i]>=input_shape[i]:
            return False
    return True

def remove_noise(voxel_data, output = False):

    # return voxel_data

    for k in non_place_items:
        non_place_ones = np.where(voxel_data == k)
        if non_place_ones[0].size!=0:
            for j in range(non_place_ones[0].size):
                j = (non_place_ones[0][j],non_place_ones[1][j],non_place_ones[2][j])
                flag = False
                for i in range(4):
                    # print(DXYZ[i])
                    j += DXYZ[i]
                    j = tuple(j)
                    # print(j)
                    if check_bound(j,voxel_data.shape):
                        if voxel_data[j]>0 and voxel_data[j] not in non_place_items:
                            flag = True
                            break
                if not flag:
                    voxel_data[j]=0

    ones = np.where(voxel_data>0,1,0)
    labeled = measure.label(ones,connectivity=1)
    total_seg = np.max(labeled)

    # print(total_seg)
    for i in range(1,total_seg+1):
        ordered_seg = np.where(labeled==i,1,0)
        temp_sum = np.sum(ordered_seg)
        if temp_sum < MIN_SEG_SIZE:
            labeled -= ordered_seg*i

    if output:
        for j in range(voxel_data.shape[0]):
            np.savetxt('combined_{}.txt'.format(j), labeled[j], fmt = '%d')
    # new_ones = np.where(labeled>0,1,0)
    # if(new_ones.sum()<ones.sum()):
    #     print("{} --> {}".format(ones.sum(),new_ones.sum()))

    voxel_data = np.where(labeled>0,voxel_data,0)
    # print(total_seg)
    # print(label(voxel_data))
    return voxel_data

def remove_empty(voxel_data):
    for i in range(H):
        if np.any(voxel_data[0]):
            return voxel_data
        else:
            voxel_data = np.delete(voxel_data,0,0)
            append_arr = np.zeros((1,W,L))
            voxel_data = np.concatenate((voxel_data,append_arr),axis = 0)
    return voxel_data

def convert_seq(seq,ori_vox,start_height=0):
    new_seq = []
    # ori_vox = np.load(ori_vox_path)
    # ori_vox = np.transpose(ori_vox,(0,2,1))
    for i in seq:
        if ori_vox[tuple(i)] >0:
            block_type = ori_vox[tuple(i)]
            new_seq.append([i[0],i[1],i[2],block_type])
    return new_seq



def vox2seq(vox_path = "./results/voxel",model_path = "./models/vox2seq.pt",save_path = "./results/seq", max_step = 1000):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cnn = ResNet3D(BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512], num_output=32*32*32+1)
    cnn.load_state_dict(torch.load(model_path))
    cnn.to('cpu')
    cnn.eval()

    vox_files = os.listdir(vox_path)
    npy_files = [file for file in vox_files if file.endswith(".npy")]

    for j in npy_files:
        v_pth = os.path.join(vox_path,j)
        ori_vox = np.load(v_pth)
        assert(ori_vox.shape==(H,W,L))
        ori_vox = remove_empty(ori_vox)
        ori_vox = remove_noise(ori_vox)
        ori_vox = np.transpose(ori_vox,(0,2,1))
        vox = np.where(ori_vox>0,1,0)
        max_step = vox.sum()-10
        vox = torch.from_numpy(vox).float().to('cpu')
        # print(vox)
        output = cnn.inference(vox,max_step)

        seq = []
        for i in output:
            seq.append(i.detach().cpu().numpy())
        # seq = np.array(seq)

        new_seq = convert_seq(seq,ori_vox)
        with open(os.path.join(save_path,os.path.splitext(j)[0]+'.pkl'), 'wb') as f:
            pickle.dump(new_seq, f)
    



if __name__ == '__main__':
    torch.set_printoptions(profile="full")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./vox2seq.pt')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_step', type=int, default=500)
    args = parser.parse_args()

    # main(args)
    vox2seq()
