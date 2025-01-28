# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit


import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchprofile import profile_macs

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.function import testval, test
from utils.utils import create_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = model = models.pidnet.get_seg_model(config, imgnet_pretrained=True)

    model = model.cuda()

    model.eval()
    input = torch.randn(1, 3, 1024, 1024).cuda()
    macs = profile_macs(model, input)

    print('MACs: %d' % macs)

    import time
    start_time = time.time()
    output = model(input)
    end_time = time.time()

    print('Time: %f' % (end_time - start_time))

    # number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters: %d' % num_params)


if __name__ == '__main__':
    main()
