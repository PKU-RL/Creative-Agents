
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from dataset import VoxelDataset
from model import ResNet3D, BasicBlock


class LightningTrainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.cnn = ResNet3D(BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512], num_output=32*3 + 1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, voxel):
        return self.cnn(voxel)

    def setup(self, stage):
        self.dataset = VoxelDataset(self.args.dataset_dir)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [int(len(self.dataset) * 0.8), len(self.dataset) - int(len(self.dataset) * 0.8)])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def training_step(self, batch, batch_idx):
        voxel, position = batch
        output = self(voxel)
        loss = self.criterion(output, position)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        voxel, position = batch
        output = self(voxel)
        loss = self.criterion(output, position)
        self.log('val_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        acc = self._compute_accuracy(output, position)
        self.log('val_acc', acc, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)


        return output

    def _compute_accuracy(self, output, position):
        return (output.argmax(dim=-1) == position).sum().item() / output.size(0)

    def get_model_callback(self):
        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=f"./ckpt/{self.args.name}", filename='{epoch}-{val_acc:.2f}', monitor='val_acc', mode='max', save_top_k=3)
        return [checkpoint]


def main(args):
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(args.seed)

    # csv_logger = pl.loggers.CSVLogger(save_dir="./log")
    wandb_logger = WandbLogger(entity=args.entity, project='voxel2position', group=args.group)
    wandb_logger.experiment.config.update({"lr": args.lr, "seed": args.seed, "batch_size": args.batch_size, "hidden_dim": args.hidden_dim, "max_length": args.max_length})

    model = LightningTrainer(args)
    trainer = Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        accelerator='gpu',
        devices=8,
        log_every_n_steps=1,
        callbacks=model.get_model_callback()
    )
    if args.resume_from_checkpoint is None:
        trainer.fit(model)
    else:
        trainer.fit(model, ckpt_path=args.resume_from_checkpoint,)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='exp1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--dataset_dir', type=str, default='./dataset')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    args = parser.parse_args()
    main(args)
