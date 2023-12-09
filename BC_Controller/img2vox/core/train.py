# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os
import random
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.test_mc import test_net_mc_RGB

# from models_MC_RGB.encoderM import EncoderM
# from models_MC_RGB.decoderM import DecoderM
from models_MC_RGB.refinerM import RefinerM
from models_MC_RGB.mergerM import MergerM

from models_MC_RGB.encoderM import EncoderM_512
from models_MC_RGB.decoderM import DecoderM_512
from models_MC_RGB.encoderM import EncoderM_224
from models_MC_RGB.decoderM import DecoderM_224

import numpy as np
import PIL



OUTPUT_FEQ = 1

def my_collate_fn(batch):
    data = [torch.tensor(np.array(item)) if isinstance(
        item, PIL.Image.Image) else item for item in batch]
    return torch.utils.data.dataloader.default_collate(data)


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    # train_transforms = utils.data_transforms.Compose([
    #     utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
    #     utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
    #     utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
    #     utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
    #     utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
    #     utils.data_transforms.RandomFlip(),
    #     utils.data_transforms.RandomPermuteRGB(),
    #     utils.data_transforms.ToTensor(),
    # ])
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        # utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        # utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])
    # val_transforms = utils.data_transforms.Compose([
    #     utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
    #     utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
    #     utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
    #     utils.data_transforms.ToTensor(),
    # ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        # utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        # utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])
    VOX_PATH = cfg.DATASETS.BC_VOXEL_PATH
    IMAGE_PATH = cfg.DATASETS.BC_IMG_PATH
    # Set up data loader
    if cfg.DATASET.TRAIN_DATASET == "MC_RGB":
        train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](
            VOX_PATH, IMAGE_PATH, True, train_transforms)
        val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](
            VOX_PATH, IMAGE_PATH, False, val_transforms)
        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader,
                                                        batch_size=cfg.CONST.BATCH_SIZE,
                                                        num_workers=cfg.TRAIN.NUM_WORKER,
                                                        pin_memory=True,
                                                        shuffle=True,
                                                        drop_last=True,
                                                        collate_fn=my_collate_fn,
                                                        )
        val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader,
                                                      batch_size=1,
                                                      num_workers=2,
                                                      pin_memory=True,
                                                      shuffle=False,
                                                      collate_fn=my_collate_fn,
                                                      )
        # print("HERE: val_dataset_loader_len:{}".format(len(val_data_loader)))
    else:
        train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](
            cfg)
        val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](
            cfg)
        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
            batch_size=cfg.CONST.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKER,
            pin_memory=True,
            shuffle=True,
            drop_last=True)
        val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            shuffle=False)
    # Set up networks
    if cfg.CONST.IMG_W ==224:
        encoder = EncoderM_224(cfg)
        decoderR = DecoderM_224(cfg)
        decoderG = DecoderM_224(cfg)
        decoderB = DecoderM_224(cfg)
        decoderS = DecoderM_224(cfg)
    else:
        encoder = EncoderM_512(cfg)
        decoderR = DecoderM_512(cfg)
        decoderG = DecoderM_512(cfg)
        decoderB = DecoderM_512(cfg)
        decoderS = DecoderM_512(cfg)

    refinerB = None
    refinerG = None
    refinerR = None
    refinerS = None

    if cfg.NETWORK.USE_REFINER:
        refinerB = RefinerM(cfg)
        refinerG = RefinerM(cfg)
        refinerR = RefinerM(cfg)
        refinerS = RefinerM(cfg)
    # merger = MergerM(cfg)
    print('[DEBUG] %s Parameters in Encoder: %d.' %
            (dt.now(), utils.network_utils.count_parameters(encoder)))
    print('[DEBUG] %s Parameters in DecoderR: %d.' %
            (dt.now(), utils.network_utils.count_parameters(decoderR)))
    if cfg.NETWORK.USE_REFINER:
        print('[DEBUG] %s Parameters in RefinerR: %d.' %
                (dt.now(), utils.network_utils.count_parameters(refinerR)))
    # print('[DEBUG] %s Parameters in Merger: %d.' % (dt.now(), utils.network_utils.count_parameters(merger)))

    # Initialize weights of networks
    encoder.apply(utils.network_utils.init_weights)
    decoderR.apply(utils.network_utils.init_weights)

    decoderG.apply(utils.network_utils.init_weights)
    decoderB.apply(utils.network_utils.init_weights)

    decoderS.apply(utils.network_utils.init_weights)
    if cfg.NETWORK.USE_REFINER:
        refinerB.apply(utils.network_utils.init_weights)
        refinerG.apply(utils.network_utils.init_weights)
        refinerR.apply(utils.network_utils.init_weights)
        refinerS.apply(utils.network_utils.init_weights)
    # merger.apply(utils.network_utils.init_weights)

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                            lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                            betas=cfg.TRAIN.BETAS)
        decoderR_solver = torch.optim.Adam(decoderR.parameters(),
                                            lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                            betas=cfg.TRAIN.BETAS)

        decoderG_solver = torch.optim.Adam(decoderG.parameters(),
                                            lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                            betas=cfg.TRAIN.BETAS)

        decoderB_solver = torch.optim.Adam(decoderB.parameters(),
                                            lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                            betas=cfg.TRAIN.BETAS)

        decoderS_solver = torch.optim.Adam(decoderS.parameters(),
                                            lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                            betas=cfg.TRAIN.BETAS)
        
        refinerB_solver = None
        refinerG_solver = None
        refinerR_solver = None
        refinerS_solver = None
        if cfg.NETWORK.USE_REFINER:
            refinerB_solver = torch.optim.Adam(refinerB.parameters(),
                                                lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                                betas=cfg.TRAIN.BETAS)
            refinerG_solver = torch.optim.Adam(refinerG.parameters(),
                                                lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                                betas=cfg.TRAIN.BETAS)
            refinerR_solver = torch.optim.Adam(refinerR.parameters(),
                                                lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                                betas=cfg.TRAIN.BETAS)
            refinerS_solver = torch.optim.Adam(refinerS.parameters(),
                                                lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                                betas=cfg.TRAIN.BETAS)
        # merger_solver = torch.optim.Adam(merger.parameters(), lr=cfg.TRAIN.MERGER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    # elif cfg.TRAIN.POLICY == 'sgd':
    #     encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()),
    #                                     lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
    #                                     momentum=cfg.TRAIN.MOMENTUM)
    #     decoder_solver = torch.optim.SGD(decoder.parameters(),
    #                                     lr=cfg.TRAIN.DECODER_LEARNING_RATE,
    #                                     momentum=cfg.TRAIN.MOMENTUM)
    #     refiner_solver = torch.optim.SGD(refiner.parameters(),
    #                                     lr=cfg.TRAIN.REFINER_LEARNING_RATE,
    #                                     momentum=cfg.TRAIN.MOMENTUM)
    #     # merger_solver = torch.optim.SGD(merger.parameters(),
    #     #                                 lr=cfg.TRAIN.MERGER_LEARNING_RATE,
    #     #                                 momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' %
                        (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver,
                                                                milestones=cfg.TRAIN.ENCODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    decoderR_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoderR_solver,
                                                                    milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                    gamma=cfg.TRAIN.GAMMA)

    decoderG_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoderG_solver,
                                                                    milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                    gamma=cfg.TRAIN.GAMMA)

    decoderB_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoderB_solver,
                                                                    milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                    gamma=cfg.TRAIN.GAMMA)

    decoderS_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoderS_solver,
                                                                    milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                    gamma=cfg.TRAIN.GAMMA)
    if cfg.NETWORK.USE_REFINER:
        refinerB_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refinerB_solver,
                                                                        milestones=cfg.TRAIN.REFINER_LR_MILESTONES,
                                                                        gamma=cfg.TRAIN.GAMMA)
        refinerG_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refinerG_solver,
                                                                        milestones=cfg.TRAIN.REFINER_LR_MILESTONES,
                                                                        gamma=cfg.TRAIN.GAMMA)
        refinerR_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refinerR_solver,
                                                                        milestones=cfg.TRAIN.REFINER_LR_MILESTONES,
                                                                        gamma=cfg.TRAIN.GAMMA)
        refinerS_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refinerS_solver,
                                                                        milestones=cfg.TRAIN.REFINER_LR_MILESTONES,
                                                                        gamma=cfg.TRAIN.GAMMA)
    # merger_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(merger_solver,
    #                                                         milestones=cfg.TRAIN.MERGER_LR_MILESTONES,
    #                                                         gamma=cfg.TRAIN.GAMMA)

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoderR = torch.nn.DataParallel(decoderR).cuda()
        decoderG = torch.nn.DataParallel(decoderG).cuda()
        decoderB = torch.nn.DataParallel(decoderB).cuda()
        decoderS = torch.nn.DataParallel(decoderS).cuda()
        if cfg.NETWORK.USE_REFINER:
            refinerB = torch.nn.DataParallel(refinerB).cuda()
            refinerR = torch.nn.DataParallel(refinerR).cuda()
            refinerG = torch.nn.DataParallel(refinerG).cuda()
            refinerS = torch.nn.DataParallel(refinerS).cuda()
        # merger = torch.nn.DataParallel(merger).cuda()

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        print('[INFO] %s Recovering from %s ...' %
                (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoderR.load_state_dict(checkpoint['decoderR_state_dict'])
        decoderG.load_state_dict(checkpoint['decoderG_state_dict'])
        decoderB.load_state_dict(checkpoint['decoderB_state_dict'])
        decoderS.load_state_dict(checkpoint['decoderS_state_dict'])

        if cfg.NETWORK.USE_REFINER:
            refinerR.load_state_dict(checkpoint['refinerR_state_dict'])
            refinerG.load_state_dict(checkpoint['refinerG_state_dict'])
            refinerB.load_state_dict(checkpoint['refinerB_state_dict'])
            refinerS.load_state_dict(checkpoint['refinerS_state_dict'])

        # if cfg.NETWORK.USE_MERGER:
        #     merger.load_state_dict(checkpoint['merger_state_dict'])

        print('[INFO] %s Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
                (dt.now(), init_epoch, best_iou, best_epoch))

    # Summary writer for TensorBoard
    if not os.path.exists(cfg.DIR.OUT_PATH):
        os.mkdir(cfg.DIR.OUT_PATH)
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)

    log_dir = output_dir % 'logs'
    ckpt_dir = output_dir % 'checkpoints'
    img_dir = output_dir % 'images'
    # if not os.path.exists(img_dir):
    #     os.mkdir(img_dir)
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'test'))

    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        encoderR_losses = utils.network_utils.AverageMeter()
        encoderG_losses = utils.network_utils.AverageMeter()
        encoderB_losses = utils.network_utils.AverageMeter()
        encoderS_losses = utils.network_utils.AverageMeter()
        if cfg.NETWORK.USE_REFINER:
            refinerR_losses = utils.network_utils.AverageMeter()
            refinerG_losses = utils.network_utils.AverageMeter()
            refinerB_losses = utils.network_utils.AverageMeter()
            refinerS_losses = utils.network_utils.AverageMeter()

        # switch models to training mode
        encoder.train()
        decoderR.train()
        decoderG.train()
        decoderB.train()
        decoderS.train()
        # merger.train()
        if cfg.NETWORK.USE_REFINER:
            refinerR.train()
            refinerG.train()
            refinerB.train()
            refinerS.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (rendering_images,
                        ground_truth_volumes) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)

            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(
                rendering_images)
            ground_truth_volumes = utils.network_utils.var_or_cuda(
                ground_truth_volumes)
            # print("HERE: {}".format(ground_truth_volumes))

            # Train the encoder, decoder, refiner, and merger
            image_features = encoder(rendering_images)
            shape_masks = ground_truth_volumes.to(torch.float32)[
                :, :, :, :, 3]

            raw_features, generated_volumes = decoderR(image_features)

            # generated_volumes = torch.mean(generated_volumes, dim=1)
            # print(generated_volumes.shape)
            generated_volumes = generated_volumes.to(
                torch.float32)*shape_masks
            # print(ground_truth_volumes.shape)
            ground_truth_volumesR = ground_truth_volumes.to(torch.float32)[
                :, :, :, :, 0]
            encoderR_loss = bce_loss(
                generated_volumes, ground_truth_volumesR) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volumes = refinerR(generated_volumes)
                refinerR_loss = bce_loss(
                    generated_volumes, ground_truth_volumesR) * 10
            else:
                refinerR_loss = encoderR_loss

            raw_features, generated_volumes = decoderG(image_features)

            # generated_volumes = torch.mean(generated_volumes, dim=1)
            generated_volumes = generated_volumes.to(
                torch.float32)*shape_masks
            ground_truth_volumesG = ground_truth_volumes.to(torch.float32)[
                :, :, :, :, 1]
            encoderG_loss = bce_loss(
                generated_volumes, ground_truth_volumesG) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volumes = refinerG(generated_volumes)
                refinerG_loss = bce_loss(
                    generated_volumes, ground_truth_volumesG) * 10
            else:
                refinerG_loss = encoderG_loss

            raw_features, generated_volumes = decoderB(image_features)

            # generated_volumes = torch.mean(generated_volumes, dim=1)
            generated_volumes = generated_volumes.to(
                torch.float32)*shape_masks
            ground_truth_volumesB = ground_truth_volumes.to(torch.float32)[
                :, :, :, :, 2]
            encoderB_loss = bce_loss(
                generated_volumes, ground_truth_volumesB) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volumes = refinerB(generated_volumes)
                refinerB_loss = bce_loss(
                    generated_volumes, ground_truth_volumesB) * 10
            else:
                refinerB_loss = encoderB_loss

            raw_features, generated_volumes = decoderS(image_features)

            # generated_volumes = torch.mean(generated_volumes, dim=1)
            generated_volumes = generated_volumes.to(torch.float32)
            ground_truth_volumesS = ground_truth_volumes.to(torch.float32)[
                :, :, :, :, 3]
            encoderS_loss = bce_loss(
                generated_volumes, ground_truth_volumesS) * 10
            
            # occ_s_loss
            # temp_generated_volumes = torch.where(generated_volumes>0.5, 1, 0)
            # encoderS_loss += ((ground_truth_volumesS.sum((1,2,3)) - temp_generated_volumes.sum((1,2,3)))*(ground_truth_volumesS.sum((1,2,3)) - temp_generated_volumes.sum((1,2,3)))).sum()/(32*32*32*10000)
            # print("loss")
            # print(((ground_truth_volumesS.sum((1,2,3)) - generated_volumes.sum((1,2,3)))*(ground_truth_volumesS.sum((1,2,3)) - generated_volumes.sum((1,2,3)))).sum()/(32*32*32*10000))
            # # print()
            # print(encoderS_loss)

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volumes = refinerS(generated_volumes)
                refinerS_loss = bce_loss(
                    generated_volumes, ground_truth_volumesS) * 10
            else:
                refinerS_loss = encoderS_loss

            # Gradient decent
            encoder.zero_grad()
            decoderR.zero_grad()
            decoderG.zero_grad()
            decoderB.zero_grad()
            decoderS.zero_grad()
            if cfg.NETWORK.USE_REFINER:
                refinerR.zero_grad()
                refinerG.zero_grad()
                refinerB.zero_grad()
                refinerS.zero_grad()

            # merger.zero_grad()

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                encoder_loss = 0.8*encoderR_loss + 0.8*encoderG_loss + \
                    0.8*encoderB_loss + 1.6*encoderS_loss
                encoder_loss.backward(retain_graph=True)
                refiner_loss = 0.8*refinerR_loss + 0.8*refinerG_loss + \
                    0.8*refinerB_loss + 1.6*refinerS_loss
                refiner_loss.backward()
            else:
                encoder_loss = 0.8*encoderR_loss + 0.8*encoderG_loss + \
                    0.8*encoderB_loss + 1.6*encoderS_loss
                encoder_loss.backward()

            encoder_solver.step()
            decoderR_solver.step()

            decoderG_solver.step()

            decoderB_solver.step()

            decoderS_solver.step()
            if cfg.NETWORK.USE_REFINER:

                refinerB_solver.step()
                refinerG_solver.step()
                refinerR_solver.step()
                refinerS_solver.step()
            # merger_solver.step()

            # Append loss to average metrics
            encoderR_losses.update(encoderR_loss.item())

            encoderG_losses.update(encoderG_loss.item())

            encoderB_losses.update(encoderB_loss.item())
            encoderS_losses.update(encoderS_loss.item())
            
            if cfg.NETWORK.USE_REFINER:
                refinerB_losses.update(refinerB_loss.item())
                refinerG_losses.update(refinerG_loss.item())
                refinerR_losses.update(refinerR_loss.item())

                refinerS_losses.update(refinerS_loss.item())
            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar(
                'EncoderDecoder/BatchLoss', encoder_loss.item(), n_itr)
            if cfg.NETWORK.USE_REFINER:
                train_writer.add_scalar(
                    'Refiner/BatchLoss', refiner_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            if (batch_idx + 1) % OUTPUT_FEQ == 0:
                if not cfg.NETWORK.USE_REFINER:
                    print(
                        '[INFO] %s [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) EDLoss = %.4f ED_S_Loss = %.4f '
                        % (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time.val,
                        data_time.val, encoder_loss.item(), encoderS_loss.item()))
                else:
                    print(
                                '[INFO] %s [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) EDLoss = %.4f RLoss = %.4f ED_S_Loss = %.4f R_S_Loss = %.4f'
                                % (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time.val,
                                data_time.val, encoder_loss.item(), refiner_loss.item(), encoderS_loss.item(), refinerS_loss.item()))

        # Append epoch loss to TensorBoard
        train_writer.add_scalar(
            'EncoderDecoder/EpochLoss', encoderS_losses.avg, epoch_idx + 1)
        if cfg.NETWORK.USE_REFINER:
            train_writer.add_scalar(
                'Refiner/EpochLoss', refinerS_losses.avg, epoch_idx + 1)

        train_writer.add_scalar('EncoderDecoder/Epoch_RGB_Loss', encoderR_losses.avg +
                                encoderG_losses.avg + encoderB_losses.avg, epoch_idx + 1)
        if cfg.NETWORK.USE_REFINER:
            train_writer.add_scalar('Refiner/Epoch_RGB_Loss', refinerR_losses.avg +
                                    refinerG_losses.avg + refinerB_losses.avg, epoch_idx + 1)

        # Adjust learning rate
        encoder_lr_scheduler.step()
        decoderR_lr_scheduler.step()
        
        decoderG_lr_scheduler.step()
        
        decoderB_lr_scheduler.step()
        
        decoderS_lr_scheduler.step()
        if cfg.NETWORK.USE_REFINER:
            refinerB_lr_scheduler.step()
            refinerG_lr_scheduler.step()
            refinerR_lr_scheduler.step()
            refinerS_lr_scheduler.step()
        # merger_lr_scheduler.step()

        # Tick / tock
        epoch_end_time = time()
        if cfg.NETWORK.USE_REFINER:
            print('[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) EDLoss = %.4f RLoss = %.4f' %
                (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, encoderS_losses.avg,
                refinerS_losses.avg))
        else:
            print('[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) EDLoss = %.4f' %
                (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, encoderS_losses.avg
                ))

        # Update Rendering Views
        if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
            n_views_rendering = random.randint(
                1, cfg.CONST.N_VIEWS_RENDERING)
            train_data_loader.dataset.set_n_views_rendering(
                n_views_rendering)
            print('[INFO] %s Epoch [%d/%d] Update #RenderingViews to %d' %
                    (dt.now(), epoch_idx + 2, cfg.TRAIN.NUM_EPOCHES, n_views_rendering))

        # Validate the training models
        # iou = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, encoder, decoder, refiner, merger)
        if cfg.NETWORK.USE_REFINER:
            test_net_mc_RGB(cfg, False, epoch_idx + 1, img_dir, val_data_loader, val_writer, encoder,
                            decoderR, decoderG, decoderB, decoderS, refinerR, refinerG, refinerB, refinerS)
        else:
            test_net_mc_RGB(cfg, False, epoch_idx + 1, img_dir, val_data_loader, val_writer, encoder,
                            decoderR, decoderG, decoderB, decoderS, )

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            if cfg.NETWORK.USE_REFINER:
                utils.network_utils.save_checkpointsRGB(cfg, os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth' % (epoch_idx + 1)),
                                                        epoch_idx + 1, encoder, encoder_solver,
                                                        decoderR, decoderR_solver,
                                                        refinerR, refinerR_solver,
                                                        decoderG, decoderG_solver,
                                                        refinerG, refinerG_solver,
                                                        decoderB, decoderB_solver,
                                                        refinerB, refinerB_solver,
                                                        decoderS, decoderS_solver,
                                                        refinerS, refinerS_solver,
                                                        # merger,
                                                        #   merger_solver,
                                                        best_iou, best_epoch)
            else:
                utils.network_utils.save_checkpointsRGB(cfg, os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth' % (epoch_idx + 1)),
                                                        epoch_idx + 1, encoder, encoder_solver,
                                                        decoderR, decoderR_solver,
                                                        refinerR, refinerR_solver,
                                                        decoderG, decoderG_solver,
                                                        refinerG, refinerG_solver,
                                                        decoderB, decoderB_solver,
                                                        refinerB, refinerB_solver,
                                                        decoderS, decoderS_solver,
                                                        refinerS, refinerS_solver,
                                                        # merger,
                                                        #   merger_solver,
                                                        best_iou, best_epoch)
        # if iou > best_iou:
        #     if not os.path.exists(ckpt_dir):
        #         os.makedirs(ckpt_dir)

        #     best_iou = iou
        #     best_epoch = epoch_idx + 1
        #     utils.network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'best-ckpt.pth'), epoch_idx + 1, encoder,
        #                                          encoder_solver, decoder, decoder_solver, refiner, refiner_solver,
        #                                          merger, merger_solver, best_iou, best_epoch)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()