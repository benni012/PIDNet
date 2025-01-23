# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate

def train_adapt(config, epoch, num_epoch, epoch_iters, base_lr,
               num_iters, trainloader, optimizer, model, writer_dict, targetloader, optimizer_dis, model_dis):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    avg_dis_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    targetloader_iter = iter(targetloader)

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, bd_gts, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()

        # get target data
        try:
            batch_target = next(targetloader_iter)
        except:
            targetloader_iter = iter(targetloader)
            batch_target = next(targetloader_iter)

        images_target, _, _ = batch_target
        images_target = images_target.cuda()

        # generator (don't accumulate gradients for discriminator)
        # - get source and target predictions
        # - loss = seg_loss (src) + adv_loss (src + target)
        # discriminator (accumulate)
        # - get source and target predictions
        # - loss = -adv_loss (src + target)

        def get_pred(model, images):
            outputs = model.module.model(images)
            h, w = labels.size(1), labels.size(2)
            for i in range(len(outputs)):
                outputs[i] = F.interpolate(outputs[i], size=( h,w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
            labels_logits = outputs[1]
            # labels_prob = F.softmax(labels_logits, dim=1)
            # max_probs, labels = torch.max(labels_prob, dim=1)
            bd = outputs[2][:,0]
            return labels_logits, bd, outputs

        # GENERATOR PHASE
        # disable grad for discriminator
        for param in model_dis.parameters():
            param.requires_grad = False

        # get source and target predictions
        pred_src, _, outputs_src = get_pred(model, images)
        pred_tgt, _, _ = get_pred(model, images_target)

        # discriminate
        pred_src_adv = model_dis(pred_src)
        pred_tgt_adv = model_dis(pred_tgt)

        # loss = seg_loss (src) + adv_loss (src + target)
        loss_adv_src = F.binary_cross_entropy_with_logits(pred_src_adv, torch.ones_like(pred_src_adv))
        loss_adv_tgt = F.binary_cross_entropy_with_logits(pred_tgt_adv, torch.zeros_like(pred_tgt_adv))

        # print("Discriminator loss: " + loss_adv_src.item() + loss_adv_tgt.item())
        print(f"Discriminator loss: {loss_adv_src.item() + loss_adv_tgt.item()}")


        # losses, _, acc, loss_list = model(images, labels, bd_gts)
        losses, _, acc, loss_list = model.module.get_loss(outputs_src, labels, bd_gts)
        lamb = 0.001
        loss = losses.mean() + lamb * (loss_adv_src + loss_adv_tgt)
        acc = acc.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # DISCRIMINATOR PHASE
        # enable grad for discriminator
        for param in model_dis.parameters():
            param.requires_grad = True

        # get source and target predictions
        pred_src, _, _ = get_pred(model, images)
        pred_tgt, _, _ = get_pred(model, images_target)

        # discriminate
        pred_src_adv = model_dis(pred_src)
        pred_tgt_adv = model_dis(pred_tgt)

        # loss = -adv_loss (src + target)
        loss_adv_src = F.binary_cross_entropy_with_logits(pred_src_adv, torch.zeros_like(pred_src_adv))
        loss_adv_tgt = F.binary_cross_entropy_with_logits(pred_tgt_adv, torch.ones_like(pred_tgt_adv))
        loss_dis = loss_adv_src + loss_adv_tgt

        model_dis.zero_grad()
        loss_dis.backward()
        optimizer_dis.step()


        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_bce_loss.update(loss_list[1].mean().item())
        avg_dis_loss.update(loss_dis.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter + cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}, BCE loss: {:.6f}, SB loss: {:.6f}, Discriminator loss: {:.6f}'.format(
                epoch, num_epoch, i_iter, epoch_iters,
                batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average(),
                ave_loss.average() - avg_sem_loss.average() - avg_bce_loss.average(), avg_dis_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1


def train_dacs(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict, targetloader):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    targetloader_iter = iter(targetloader)

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, bd_gts, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()

        # get target data
        try:
            batch_target = next(targetloader_iter)
        except:
            targetloader_iter = iter(targetloader)
            batch_target = next(targetloader_iter)

        images_target, _, _ = batch_target
        images_target = images_target.cuda()
        outputs = model.module.model(images_target)
        h, w = labels.size(1), labels.size(2)
        for i in range(len(outputs)):
            outputs[i] = F.interpolate(outputs[i], size=( h,w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        labels_target_logits = outputs[1]
        labels_target_prob = F.softmax(labels_target_logits, dim=1)
        max_probs, labels_target = torch.max(labels_target_prob, dim=1)
        bd_target = outputs[2][:,0]

        # go over batch dim
        images_mix, labels_mix = images_target.clone().detach(), labels_target.clone().detach()
        bd_mix = bd_target.clone().detach()

        # Ottieni tutte le classi per ogni immagine del batch
        all_classes = [torch.unique(lbl, sorted=True) for lbl in labels]
        all_classes = [cls[1:] if cls[0] == config.TRAIN.IGNORE_LABEL else cls for cls in all_classes]

        # Seleziona metà delle classi in modo casuale per ogni immagine
        selected_classes = [cls[torch.randperm(len(cls))[:(len(cls) + 1) // 2]] for cls in all_classes]

        # Inizializza una maschera globale per tutto il batch
        batch_mask = torch.zeros_like(labels, dtype=torch.bool)

        # Crea la maschera cumulativa per tutte le immagini
        for i, classes in enumerate(selected_classes):
            for c in classes:
                batch_mask[i] |= labels[i] == c

        # Applica la maschera su tutto il batch
        images_mix = torch.where(batch_mask.unsqueeze(1), images, images_mix)
        labels_mix = torch.where(batch_mask, labels, labels_mix)
        bd_mix = torch.where(batch_mask, bd_gts, bd_mix)

        # import matplotlib.pyplot as plt
        # import matplotlib
        #
        # for b in range(images.size(0)):
        #     fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        #
        #     axes[0, 0].imshow(matplotlib.colors.Normalize(clip=False)(images[b].permute(1, 2, 0).cpu().numpy()))
        #     axes[0, 0].set_title("Original Image")
        #     axes[0, 0].axis("off")
        #
        #     axes[0, 1].imshow(matplotlib.colors.Normalize(clip=False)(images_target[b].permute(1, 2, 0).cpu().numpy()))
        #     axes[0, 1].set_title("Target Image")
        #     axes[0, 1].axis("off")
        #
        #     axes[0, 2].imshow(matplotlib.colors.Normalize(clip=False)(images_mix[b].permute(1, 2, 0).cpu().numpy()))
        #     axes[0, 2].set_title("Mixed Image")
        #     axes[0, 2].axis("off")
        #
        #     axes[1, 0].imshow(labels[b].cpu().numpy(), cmap="gray", vmin=0, vmax=7)
        #     axes[1, 0].set_title("Original Mask")
        #     axes[1, 0].axis("off")
        #
        #     axes[1, 1].imshow(labels_target[b].cpu().numpy(), cmap="gray", vmin=0, vmax=7)
        #     axes[1, 1].set_title("Target Mask")
        #     axes[1, 1].axis("off")
        #
        #     axes[1, 2].imshow(labels_mix[b].cpu().numpy(), cmap="gray", vmin=0, vmax=7)
        #     axes[1, 2].set_title("Mixed Mask")
        #     axes[1, 2].axis("off")
        #
        #     # Aggiungi spazio tra i subplot
        #     plt.tight_layout()
        #
        #     # Mostra la figura
        #     plt.show()

        # TODO other confidence
        lamb = torch.sum(max_probs.ge(0.968).long() == 1, dim=(1,2)) / (max_probs.size(1) * max_probs.size(2))
        losses, _, acc, loss_list = model(images, labels, bd_gts)
        mix_losses, _, mix_acc, mix_loss_list = model(images_mix, labels_mix, bd_mix)
        print(mix_losses.shape)
        loss = losses.mean() + (mix_losses[0].mean(dim=(1,2))*lamb).mean()
        acc = acc.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_bce_loss.update(loss_list[1].mean().item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter + cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}, BCE loss: {:.6f}, SB loss: {:.6f}'.format(
                epoch, num_epoch, i_iter, epoch_iters,
                batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average(),
                ave_loss.average() - avg_sem_loss.average() - avg_bce_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, bd_gts, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()
        

        losses, _, acc, loss_list = model(images, labels, bd_gts)
        loss = losses.mean()
        acc  = acc.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_bce_loss.update(loss_list[1].mean().item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}, BCE loss: {:.6f}, SB loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average(),ave_loss.average()-avg_sem_loss.average()-avg_bce_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, bd_gts, _, _ = batch
            size = label.size()
            image = image
            label = label.long()
            bd_gts = bd_gts.float()

            losses, pred, _, _ = model(image, label, bd_gts)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            ave_loss.update(loss.item())

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        
        logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


def testval(config, test_dataset, testloader, model,
            sv_dir='./', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _, name = batch
            size = label.size()
            pred = test_dataset.single_scale_inference(config, model, image)

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='./', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                
            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
