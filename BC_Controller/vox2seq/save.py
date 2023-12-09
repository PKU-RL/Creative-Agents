import os
import argparse

import torch

from train import LightningTrainer


def main(args):
    pl_module = LightningTrainer.load_from_checkpoint(args.ckpt_path)

    # save model state dict
    torch.save(pl_module.cnn.state_dict(), args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./output/model.pt')
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()

    main(args)
