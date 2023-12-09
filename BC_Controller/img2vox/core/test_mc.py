# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data

from PIL import Image

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import utils.RGB2Item

from datetime import datetime as dt

# from models_MC_RGB.encoderM import EncoderM
# from models_MC_RGB.decoderM import DecoderM
from models_MC_RGB.refinerM import RefinerM
from models_MC_RGB.mergerM import MergerM

# from models_MC_RGB.encoderM import EncoderM_512 as EncoderM
# from models_MC_RGB.decoderM import DecoderM_512 as DecoderM

from models_MC_RGB.encoderM import EncoderM_224 as EncoderM
from models_MC_RGB.decoderM import DecoderM_224 as DecoderM

from utils.GenImage import setup_env, gen_MC_gif, gen_MC_vox

def test_net_mc_RGB(cfg,
            gen_gif = False,
             epoch_idx=-1,
             output_dir="./temp_test_RGB",
             test_data_loader=None,
             test_writer=None,
encoder = None, decoderR = None,decoderG = None,decoderB = None,decoderS = None, refinerR = None, refinerG = None,refinerB = None,refinerS = None, 
            pipline = False,
             ):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    if test_data_loader is None:
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.ToTensor(),
        ])
        if pipline:
            VOX_PATH = None
            IMAGE_PATH = cfg.DATASETS.PIP_IMG_PATH
        else:
            VOX_PATH = cfg.DATASETS.BC_VOXEL_PATH
            IMAGE_PATH = cfg.DATASETS.BC_IMG_PATH
        val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](VOX_PATH,IMAGE_PATH,False,test_transforms,pipline,cfg.DATASETS.PIP_IMG_PATH,cfg.CONST.PIP_VOX_PATH)
        test_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader,
                                                        batch_size=1,
                                                        num_workers=1,
                                                        pin_memory=True,
                                                        shuffle=False,
                                                        )
        # Set up data augmentation
    else:
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                    batch_size=1,
                                                    num_workers=1,
                                                    pin_memory=True,
                                                    shuffle=False)

    # Set up networks
    if encoder is None:
        encoder = EncoderM(cfg)

        decoderR = DecoderM(cfg)
        decoderG = DecoderM(cfg)
        decoderB = DecoderM(cfg)
        

        decoderS = DecoderM(cfg)
        if cfg.NETWORK.USE_REFINER:
            refinerB = RefinerM(cfg)
            refinerG = RefinerM(cfg)
            refinerR = RefinerM(cfg)
            refinerS = RefinerM(cfg)
        # merger = Merger(cfg)

        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoderR = torch.nn.DataParallel(decoderR).cuda()
            decoderG = torch.nn.DataParallel(decoderG).cuda()
            decoderB = torch.nn.DataParallel(decoderB).cuda()


            decoderS = torch.nn.DataParallel(decoderS).cuda()
            if cfg.NETWORK.USE_REFINER:
                refinerB = torch.nn.DataParallel(refinerB).cuda()
                refinerG = torch.nn.DataParallel(refinerG).cuda()
                refinerR = torch.nn.DataParallel(refinerR).cuda()

                refinerS = torch.nn.DataParallel(refinerS).cuda()
            # merger = torch.nn.DataParallel(merger).cuda()

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
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

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    encoderR_losses = utils.network_utils.AverageMeter()
    encoderG_losses = utils.network_utils.AverageMeter()
    encoderB_losses = utils.network_utils.AverageMeter()
    encoderS_losses = utils.network_utils.AverageMeter()
    if cfg.NETWORK.USE_REFINER:
        refinerR_losses = utils.network_utils.AverageMeter()
        refinerG_losses = utils.network_utils.AverageMeter()
        refinerB_losses = utils.network_utils.AverageMeter()
        refinerS_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoderR.eval()
    decoderG.eval()
    decoderB.eval()
    decoderS.eval()
    if cfg.NETWORK.USE_REFINER:
        refinerR.eval()
        refinerG.eval()
        refinerB.eval()
        refinerS.eval()

    for sample_idx, (rendering_images, ground_truth_volume,info) in enumerate(test_data_loader):
        # print("endering_images.shape:{}".format(rendering_images.shape))
        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)

            # Test the encoder, decoder, refiner and merger
            image_features = encoder(rendering_images)
            generated_volumes = [decoderR(image_features)[1],decoderG(image_features)[1],decoderB(image_features)[1],decoderS(image_features)[1]]

            shape_masks = generated_volumes[3][0]
            shape_masks = torch.where(shape_masks>0.5,1,0)
            generated_volume = generated_volumes[0]
            generated_volume = generated_volume.to(torch.float32)*shape_masks
            # print(ground_truth_volume.shape)
            ground_truth_volumeR = ground_truth_volume.to(torch.float32)[:,:,:,:,0]
            encoderR_loss = bce_loss(generated_volume, ground_truth_volumeR) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volume = refinerR(generated_volume)
                refinerR_loss = bce_loss(generated_volume, ground_truth_volumeR) * 10
            else:
                refinerR_loss = encoderR_loss


            generated_volume = generated_volumes[1]

            # generated_volume = torch.mean(generated_volume, dim=1)
            generated_volume = generated_volume.to(torch.float32)*shape_masks
            ground_truth_volumeG = ground_truth_volume.to(torch.float32)[:,:,:,:,1]
            encoderG_loss = bce_loss(generated_volume, ground_truth_volumeG) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volume = refinerG(generated_volume)
                refinerG_loss = bce_loss(generated_volume, ground_truth_volumeG) * 10
            else:
                refinerG_loss = encoderG_loss


            generated_volume = generated_volumes[2]

            # generated_volume = torch.mean(generated_volume, dim=1)
            generated_volume = generated_volume.to(torch.float32)*shape_masks
            ground_truth_volumeB = ground_truth_volume.to(torch.float32)[:,:,:,:,2]
            encoderB_loss = bce_loss(generated_volume, ground_truth_volumeB) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volume = refinerB(generated_volume)
                refinerB_loss = bce_loss(generated_volume, ground_truth_volumeB) * 10
            else:
                refinerB_loss = encoderB_loss


            generated_volume = generated_volumes[3]

            # generated_volume = torch.mean(generated_volume, dim=1)
            generated_volume = generated_volume.to(torch.float32)
            ground_truth_volumeS = ground_truth_volume.to(torch.float32)[:,:,:,:,3]
            encoderS_loss = bce_loss(generated_volume, ground_truth_volumeS) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volume = refinerS(generated_volume)
                refinerS_loss = bce_loss(generated_volume, ground_truth_volumeS) * 10
            else:
                refinerS_loss = encoderS_loss

            # Append loss and accuracy to average metrics
            encoderR_losses.update(encoderR_loss.item())
            encoderG_losses.update(encoderG_loss.item())
            encoderB_losses.update(encoderB_loss.item())
            encoderS_losses.update(encoderS_loss.item())
            if cfg.NETWORK.USE_REFINER:
                refinerB_losses.update(refinerB_loss.item())
                refinerG_losses.update(refinerG_loss.item())
                refinerR_losses.update(refinerR_loss.item())

                refinerS_losses.update(refinerS_loss.item())
            # encoder_losses.update(encoder_loss.item())
            # refiner_losses.update(refiner_loss.item())

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume[:,:,:,:,3])).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume[:,:,:,:,3]), 1)).float()
                sample_iou.append((intersection / union).item())

            # Append generated volumes to TensorBoard
            # if output_dir and sample_idx < 3:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            img_dir = os.path.join(output_dir,"images")
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            gif_dir = os.path.join(output_dir,"gif")
            if not os.path.exists(gif_dir):
                os.mkdir(gif_dir)
            # img_dir = output_dir % 'images'
            # Volume Visualization
            # # print("HERE:{}".format(rendering_images))

            rendering_images = np.squeeze((np.array(rendering_images.cpu())*255).astype(np.uint8))
            # print(rendering_images.shape)
            rendering_images = rendering_images.transpose(1,2,0)
            # print(rendering_images.shape)
            temp_image = Image.fromarray(rendering_images)
            temp_image = temp_image.convert('RGB')
            shape_masks = shape_masks.cpu().numpy().reshape(32,32,32)
            gvR = (generated_volumes[0].cpu().numpy()[0].reshape(32,32,32)*shape_masks).reshape(32,32,32,1)
            gvG = (generated_volumes[1].cpu().numpy()[0].reshape(32,32,32)*shape_masks).reshape(32,32,32,1)
            gvB = (generated_volumes[2].cpu().numpy()[0].reshape(32,32,32)*shape_masks).reshape(32,32,32,1)
            gvS = generated_volumes[3].cpu().numpy()[0]

            gv = np.concatenate((gvR,gvG,gvB),axis = -1)
            # print(gv.shape)mask
            gtv = ground_truth_volume.cpu().numpy()[0][:,:,:,0:3]
            gtv_s = ground_truth_volume.cpu().numpy()[0][:,:,:,3]
            # print(gtv.shape)
            # print("HERE:{}".format(gv.shape))

            if gen_gif:
                env = setup_env()
            if gen_gif:
                save_dir = gif_dir
                gtv = utils.RGB2Item.RGB2Item(gtv*255)
                gv_0 = utils.RGB2Item.RGB2Item(gv*255)

                gen_MC_gif(gv_0,env,os.path.join(save_dir,"voxels-{}.gif".format(sample_idx)),save_dir, sample_idx,gif_duration=0.8)
                gen_MC_gif(gtv,env,os.path.join(save_dir,"GroundTruth-{}.gif".format(sample_idx)),save_dir, sample_idx,gif_duration=0.8)
            
                env.close()
            else:
                save_dir = img_dir
                rendering_views = utils.binvox_visualization.get_volume_views(gvS, save_dir,
                                                                                sample_idx)
                # test_writer.add_image('Test Sample#%02d/Volume Reconstructed' % sample_idx, rendering_views, epoch_idx)
                
                rendering_views = utils.binvox_visualization.get_volume_views(gtv_s, save_dir,
                                                                            sample_idx,ground_truth=True)
            temp_image.save(os.path.join(save_dir,'input_image_{}.png'.format(sample_idx)))
            if pipline:
                save_dir = cfg.CONST.PIP_VOX_PATH
            gv_0 = utils.RGB2Item.RGB2Item(gv*255)
            if len(info) ==0:
                gen_MC_vox(gv_0,save_dir,"voxels-{}".format(sample_idx))
            else:
                gen_MC_vox(gv_0,save_dir,info[0][0])
            # test_writer.add_image('Test Sample#%02d/Volume GroundTruth' % sample_idx, rendering_views, epoch_idx)
            

            # Print sample loss and IoU
            if cfg.NETWORK.USE_REFINER:
                print('[INFO] %s Test[%d/%d] EDLoss = [%.4f,%.4f,%.4f,%.4f] RLoss = [%.4f,%.4f,%.4f,%.4f] IoU = %s' %
                    (dt.now(), sample_idx + 1, n_samples, encoderR_loss.item(),encoderG_loss.item(),encoderB_loss.item(),encoderS_loss.item(),
                    refinerR_loss.item(),refinerG_loss.item(),refinerB_loss.item(),refinerS_loss.item(), ['%.4f' % si for si in sample_iou]))
            else:
                print('[INFO] %s Test[%d/%d] EDLoss = [%.4f,%.4f,%.4f,%.4f] IoU = %s' %
                    (dt.now(), sample_idx + 1, n_samples, encoderR_loss.item(),encoderG_loss.item(),encoderB_loss.item(),encoderS_loss.item(), ['%.4f' % si for si in sample_iou]))
           

    # Output testing results

    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    print('Overall ', end='\t\t\t\t')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('EncoderDecoder/EpochLoss', encoderS_losses.avg, epoch_idx)
        if cfg.NETWORK.USE_REFINER:
            test_writer.add_scalar('Refiner/EpochLoss', refinerS_losses.avg, epoch_idx)

    return None