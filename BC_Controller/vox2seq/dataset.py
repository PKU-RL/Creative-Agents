import os
import pickle
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


class VoxelDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.data_files = os.listdir(dataset_dir)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = f"{idx}.pkl"
        with open(os.path.join(self.dataset_dir, data_file), 'rb') as f:
            data = pickle.load(f)
        
        position = torch.from_numpy(data['position']).float()
        if position[0] == -1:
            label = torch.tensor(32*32*32)
        else:
            label = position[2] + position[1] * 32 + position[0] * 32 * 32
        return torch.from_numpy(data['voxel']).float(), label.long()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./dataset')
    args = parser.parse_args()

    dataset = VoxelDataset(args.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        voxel, position = batch
        print(voxel.shape, position.shape)
        break
    