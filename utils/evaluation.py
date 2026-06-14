import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from mindspore import Tensor
from mindspore.nn.metrics import HausdorffDistance
import numpy as np
import cv2

def dice_iou_metric_binary(pred, target):
    smooth = 1e-8
    num = pred.size(0)
    m1 = pred.reshape(num, -1)
    m2 = target.reshape(num, -1)
    intersection = m1 * m2
    intersection = torch.sum(intersection, dim=1, keepdim=True)
    m1 = torch.sum(m1, dim=1, keepdim=True)
    m2 = torch.sum(m2, dim=1, keepdim=True)

    dice = (2. * intersection + smooth) / (m1 + m2 + smooth)
    iou = (intersection + smooth) / (m1 + m2 - intersection + smooth)
    return (torch.mean(dice, dim=0, keepdim=False), torch.mean(iou, dim=0, keepdim=False))

def dice_iou_metric_multiclass(input, target):
    assert input.size() == target.size()
    dice_mc = 0
    iou_mc = 0
    for channel in range(input.shape[1]):
        dice, iou = dice_iou_metric_binary(input[:, channel, ...], target[:, channel, ...])
        dice_mc += dice
        iou_mc += iou

    return dice_mc / input.shape[1], iou_mc / input.shape[1]

def hausdorff(pred, target, new_shape=(128,128)):
    num = pred.size(0)
    total_hd95 = 0
    for n in range(num):
        pred_npy = pred[n,0,:,:].cpu().numpy()
        pred_npy = pred_npy[..., np.newaxis]
        pred_npy = cv2.resize(pred_npy, new_shape, interpolation=cv2.INTER_NEAREST)
        mask_npy = target[n,0,:,:].cpu().numpy()
        mask_npy = mask_npy[..., np.newaxis]
        mask_npy = cv2.resize(mask_npy, new_shape, interpolation=cv2.INTER_NEAREST)

        x = Tensor(pred_npy)
        y = Tensor(mask_npy)
        metric = HausdorffDistance()
        metric.clear()
        metric.update(x, y, 0)
        total_hd95 += metric.eval()
    return total_hd95 / num

def hausdorff_multiclass(pred, target, new_shape=(256,256)):
    hausdorff_mc = 0
    for channel in range(pred.shape[1]):
        hausdorff_value = hausdorff(pred[:, channel:channel+1, ...], target[:, channel:channel+1, ...], new_shape=(256, 256))
        hausdorff_mc += hausdorff_value

    return hausdorff_mc / pred.shape[1]

def precision_recall(pred, target):
    num = pred.size(0)
    channel_precision = 0
    channel_recall = 0
    for n in range(num):
        channel_alone_mask_mat = target[n,0,:,:].cpu().numpy()
        channel_predict_alone_mask_mat = pred[n,0,:,:].cpu().numpy()
        channel_alone_mask_mat = channel_alone_mask_mat.reshape(-1)
        channel_predict_alone_mask_mat = channel_predict_alone_mask_mat.reshape(-1)

        channel_precision += precision_score(channel_alone_mask_mat, channel_predict_alone_mask_mat, zero_division=1)
        channel_recall += recall_score(channel_alone_mask_mat, channel_predict_alone_mask_mat, zero_division=1)

    return (channel_precision / num, channel_recall / num)

def precision_recall_multiclass(pred, target):
    precision_mc = 0
    recall_mc = 0
    for channel in range(pred.shape[1]):
        precision, recall = precision_recall(pred[:, channel:channel+1, ...], target[:, channel:channel+1, ...])
        precision_mc += precision
        recall_mc += recall

    return precision_mc / pred.shape[1], recall_mc / pred.shape[1]

def evaluate(net, dataloader, device, SSM=False):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_score = 0
    hd95_score = 0
    precision_value = 0
    recall_value = 0

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch[0], batch[1]
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            if SSM:
                mask_pred, Q_prob_collect = net(image)
            else:
                mask_pred = net(image)

            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()

            dice_batch, iou_batch = dice_iou_metric_binary(mask_pred, mask_true)
            precision_batch, recall_batch = precision_recall(mask_pred, mask_true)

            dice_score += dice_batch
            iou_score += iou_batch
            precision_value += precision_batch
            recall_value += recall_batch
            hd95_score += hausdorff(mask_pred, mask_true)

    dice = dice_score / num_val_batches
    iou = iou_score / num_val_batches
    hd95 = hd95_score / num_val_batches
    precision = precision_value / num_val_batches
    recall = recall_value / num_val_batches

    return dice, iou, '{:.4f}'.format(precision), '{:.4f}'.format(recall), '{:.4f}'.format(hd95)

def evaluate_mc(net, dataloader, device, SSM=False):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_score = 0
    hd95_score = 0
    precision_value = 0
    recall_value = 0

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round: ', unit='batch', leave=False):
        image, mask_true = batch[0], batch[1]
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            mask_true = F.one_hot(mask_true, 3).permute(0, 3, 1, 2)
            if SSM:
                mask_pred, Q_prob_collect = net(image)
            else:
                mask_pred = net(image)

            mask_pred = F.softmax(mask_pred, dim=1)
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), 3).permute(0, 3, 1, 2).float()

            dice_batch, iou_batch = dice_iou_metric_multiclass(mask_pred[:, 1:, ...], mask_true[:, 1:, ...])
            precision_batch, recall_batch = precision_recall_multiclass(mask_pred[:, 1:, ...], mask_true[:, 1:, ...])

            dice_score += dice_batch
            iou_score += iou_batch
            precision_value += precision_batch
            recall_value += recall_batch
            hd95_score += hausdorff_multiclass(mask_pred[:, 1:, ...], mask_true[:, 1:, ...])

    dice = dice_score / num_val_batches
    iou = iou_score / num_val_batches
    hd95 = hd95_score / num_val_batches
    precision = precision_value / num_val_batches
    recall = recall_value / num_val_batches

    return dice, iou, '{:.4f}'.format(precision), '{:.4f}'.format(recall), '{:.4f}'.format(hd95)

def evaluate_mc_sd(net, dataloader, device, SSM=False):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = []
    iou_score = []
    hd95_score = []
    precision_value = []
    recall_value = []

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round: ', unit='batch', leave=False):
        image, mask_true = batch[0], batch[1]
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            mask_true = F.one_hot(mask_true, 3).permute(0, 3, 1, 2)
            if SSM:
                mask_pred, Q_prob_collect = net(image)
            else:
                mask_pred = net(image)

            mask_pred = F.softmax(mask_pred, dim=1)
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), 3).permute(0, 3, 1, 2).float()

            dice_batch, iou_batch = dice_iou_metric_multiclass(mask_pred[:, 1:, ...], mask_true[:, 1:, ...])
            precision_batch, recall_batch = precision_recall_multiclass(mask_pred[:, 1:, ...], mask_true[:, 1:, ...])
            hd95_batch = hausdorff_multiclass(mask_pred[:, 1:, ...], mask_true[:, 1:, ...])

            dice_score.append(dice_batch.cpu().numpy())
            iou_score.append(iou_batch.cpu().numpy())
            precision_value.append(precision_batch)
            recall_value.append(recall_batch)
            hd95_score.append(hd95_batch)

    dice, dice_std = np.mean(dice_score), np.std(dice_score)
    iou, iou_std = np.mean(iou_score), np.std(iou_score)
    hd95, hd95_std = np.mean(hd95_score), np.std(hd95_score)
    precision, precision_std = np.mean(precision_value), np.std(precision_value)
    recall, recall_std = np.mean(recall_value), np.std(recall_value)

    return 'dice: {:.4f}, {:.4f} '.format(dice, dice_std), 'iou: {:.4f}, {:.4f}'.format(iou, iou_std), 'precision: {:.4f}, {:.4f}'.format(precision, precision_std), \
           'recall: {:.4f}, {:.4f}'.format(recall, recall_std), 'hd95: {:.4f}, {:.4f}'.format(hd95, hd95_std)

def evaluate_mc_statistic(net, dataloader, device, SSM=False, knowledge=False):
    net.eval()
    num_val_batches = len(dataloader)
    count = 0

    dice_score = []
    iou_score = []
    precision_value = []
    recall_value = []
    hd95_score = []

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round: ', unit='batch', leave=False):
        count += 1
        image, mask_true = batch[0], batch[1]
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2)
            if SSM:
                mask_pred, Q_prob_collect = net(image)
            elif knowledge:
                mask_pred = net(image)[1]
            else:
                mask_pred = net(image)

            mask_pred = F.softmax(mask_pred, dim=1)
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()

            dice_batch, iou_batch = dice_iou_metric_multiclass(mask_pred[:, 1:, ...], mask_true[:, 1:, ...])
            precision_batch, recall_batch = precision_recall_multiclass(mask_pred[:, 1:, ...], mask_true[:, 1:, ...])

            dice_score.append(dice_batch)
            iou_score.append(iou_batch)
            precision_value.append(precision_batch)
            recall_value.append(recall_batch)
            hd95_score.append(hausdorff_multiclass(mask_pred[:, 1:, ...], mask_true[:, 1:, ...]))

    number = len(dice_score)
    dice = sum(dice_score) / number
    iou = sum(iou_score) / number
    hd95 = sum(hd95_score) / number
    precision = sum(precision_value) / number
    recall = sum(recall_value) / number

    acc_slice = [item for item in dice_score if item >= 0.9]
    acc = len(acc_slice) / number

    return dice, iou, '{:.4f}'.format(precision), '{:.4f}'.format(recall), '{:.4f}'.format(hd95), '{:.4f}'.format(acc)
