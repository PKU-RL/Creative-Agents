import os
import pickle
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser


def preprocess(voxel: np.ndarray, sequence: np.ndarray, num: int, save_path: str):
    idx = 0
    origin_voxel = np.copy(voxel)

    completed_voxel = origin_voxel - voxel
    last_step_voxel = np.zeros((32, 32, 32))
    multi_channel_voxel = np.stack(
        (voxel, completed_voxel, last_step_voxel), axis=0)

    with open(os.path.join(save_path, f"{num}.pkl"), 'wb') as f:
        dic = {'voxel': multi_channel_voxel, 'position': sequence[idx]}
        pickle.dump(dic, f)

    last_position = sequence[idx]
    idx += 1
    num += 1

    for position in sequence[:-1]:
        voxel[position[0], position[1], position[2]] = 0
        completed_voxel = origin_voxel - voxel
        last_step_voxel = np.zeros((32, 32, 32))
        last_step_voxel[last_position[0],
                        last_position[1], last_position[2]] = -1
        multi_channel_voxel = np.stack(
            (voxel, completed_voxel, last_step_voxel), axis=0)

        with open(os.path.join(save_path, f"{num}.pkl"), 'wb') as f:
            dic = {'voxel': multi_channel_voxel, 'position': sequence[idx]}
            pickle.dump(dic, f)

        last_position = sequence[idx]
        idx += 1
        num += 1

    return num


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./dataset')
    parser.add_argument('--save_dir', type=str, default='./processed_dataset')
    parser.add_argument('--num_data', type=int, default=6000000)
    args = parser.parse_args()
    num = 0
    for pkl in tqdm(os.listdir(args.data_path)):
        assert pkl.endswith('pkl')
        with open(os.path.join(args.data_path, pkl), 'rb') as f:
            data = pickle.load(f)
        
        voxel = data['voxel']
        sequence = np.concatenate((data['sequence'][1:], np.array([[-1, -1, -1]])), axis=0)
        num = preprocess(voxel, sequence, num, args.save_dir)

        if num >= args.num_data:
            break
    
    # pkls = ['./0.pkl', './1.pkl']
    # for pkl in pkls:
    #     with open(pkl, 'rb') as f:
    #         data = pickle.load(f)
        
    #     voxel = data['voxel']
    #     sequence = np.concatenate((data['sequence'][1:], np.array([[-1, -1, -1]])), axis=0)
    #     print(sequence.shape)
    #     num = preprocess(voxel, sequence, num)
    #     print(num)
