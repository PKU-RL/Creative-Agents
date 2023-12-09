# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch

from datetime import datetime as dt


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def save_checkpoints(cfg, file_path, epoch_idx, encoder, encoder_solver, decoder, decoder_solver, refiner,
                     refiner_solver, merger, merger_solver, best_iou, best_epoch):
    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'best_iou': best_iou,
        'best_epoch': best_epoch,
        'encoder_state_dict': encoder.state_dict(),
        'encoder_solver_state_dict': encoder_solver.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'decoder_solver_state_dict': decoder_solver.state_dict()
    }

    if cfg.NETWORK.USE_REFINER:
        checkpoint['refiner_state_dict'] = refiner.state_dict()
        checkpoint['refiner_solver_state_dict'] = refiner_solver.state_dict()
    if cfg.NETWORK.USE_MERGER:
        checkpoint['merger_state_dict'] = merger.state_dict()
        checkpoint['merger_solver_state_dict'] = merger_solver.state_dict()

    torch.save(checkpoint, file_path)

def save_checkpointsRGB(cfg, file_path,
                                                    epoch_idx, encoder, encoder_solver,
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
                                                      best_iou, best_epoch):
    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'best_iou': best_iou,
        'best_epoch': best_epoch,
        'encoder_state_dict': encoder.state_dict(),
        'encoder_solver_state_dict': encoder_solver.state_dict(),
        'decoderR_state_dict': decoderR.state_dict(),
        'decoderR_solver_state_dict': decoderR_solver.state_dict(),

        'decoderG_state_dict': decoderG.state_dict(),
        'decoderG_solver_state_dict': decoderG_solver.state_dict(),

        'decoderB_state_dict': decoderB.state_dict(),
        'decoderB_solver_state_dict': decoderB_solver.state_dict(),

        'decoderS_state_dict': decoderS.state_dict(),
        'decoderS_solver_state_dict': decoderS_solver.state_dict()
    }

    if cfg.NETWORK.USE_REFINER:
        checkpoint['refinerR_state_dict'] = refinerR.state_dict()
        checkpoint['refinerR_solver_state_dict'] = refinerR_solver.state_dict()

        checkpoint['refinerG_state_dict'] = refinerG.state_dict()
        checkpoint['refinerG_solver_state_dict'] = refinerG_solver.state_dict()

        checkpoint['refinerB_state_dict'] = refinerB.state_dict()
        checkpoint['refinerB_solver_state_dict'] = refinerB_solver.state_dict()

        checkpoint['refinerS_state_dict'] = refinerS.state_dict()
        checkpoint['refinerS_solver_state_dict'] = refinerS_solver.state_dict()



    # if cfg.NETWORK.USE_MERGER:
    #     checkpoint['merger_state_dict'] = merger.state_dict()
    #     checkpoint['merger_solver_state_dict'] = merger_solver.state_dict()

    torch.save(checkpoint, file_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
