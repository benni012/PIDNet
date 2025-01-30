# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import matplotlib.colors
import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse
import os
import pprint

import logging
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

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


current_index = 0


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

    STOP = 99

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        blur=False,
        jitter=False,
        speedy_gonzales=config.TEST.SPEEDY_GONZALES)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    for index, batch in enumerate(tqdm(testloader)):
        image, label, _, _, name = batch

        # plt.imsave(f'vis_models/image_{index}.png', matplotlib.colors.Normalize(clip=False)(image[0].numpy().transpose(1, 2, 0)))
        # image: copy from data/loveda/val_images/{name}.png
        image_fn = f'data/loveda/val_images/{name[0]}.png'
        image = plt.imread(image_fn)
        plt.imsave(f'vis_models/{index:02d}_image.png', image)
        plt.imsave(f'vis_models/{index:02d}_gt.png', label[0].numpy(), cmap='jet')

        if index == STOP:
            break

    # build model
    model = model = models.pidnet.get_seg_model(config, imgnet_pretrained=True)

    # cycle through .pt files in vis_models/
    for file in os.listdir('vis_models/models'):
        if not file.endswith('.pt'):
            continue
        model_state_file = os.path.join('vis_models/models', file)

        logger.info('=> loading model from {}'.format(model_state_file))

        pretrained_dict = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                           if k[6:] in model_dict.keys()}
        for k, _ in pretrained_dict.items():
            logger.info(
                '=> loading {} from pretrained model'.format(k))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)



        # take 1images and run the model and show the results
        images = []
        model.eval()

        # reset testloader
        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, size, name = batch


            with torch.no_grad():
                pred = test_dataset.single_scale_inference(
                    config,
                    model,
                    image)

                pred = pred[0]
                pred = np.argmax(pred, axis=0)
                pred = np.where(label == 0, 0, pred)
                pred = pred.squeeze()
                # save prediction
                plt.imsave(f'vis_models/{index:02d}_{file}.png', pred, cmap='jet')

            if index == STOP:
                break

if __name__ == '__main__':
    main()
#
#
#
