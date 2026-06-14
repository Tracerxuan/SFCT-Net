from models.sfct_net import Stage_SSM
from torch import optim
from torch.utils.data import DataLoader
import cv2
from utils.loss import *
from utils.metrics import *
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import torch.nn.functional as F
import torch
from utils.dataloader_alone import *
import os
import torch.nn as nn
import numpy as np
import sys

sys.path.append('comparison_methods')

from medpy import metric
from medpy.metric.binary import dc, jc, hd95, sensitivity, precision

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('GPU id: ', device)
print('GPU num: ', torch.cuda.device_count())

def shift9pos(input, h_shift_unit=1, w_shift_unit=1):
    input_pd = np.pad(input, ((h_shift_unit, h_shift_unit),
                              (w_shift_unit, w_shift_unit)), mode='edge')
    input_pd = np.expand_dims(input_pd, axis=0)

    top = input_pd[:, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
    bottom = input_pd[:, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
    left = input_pd[:, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
    right = input_pd[:, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
    center = input_pd[:, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]
    bottom_right = input_pd[:, 2 * h_shift_unit:, 2 * w_shift_unit:]
    bottom_left = input_pd[:, 2 * h_shift_unit:, :-2 * w_shift_unit]
    top_right = input_pd[:, :-2 * h_shift_unit, 2 * w_shift_unit:]
    top_left = input_pd[:, :-2 * h_shift_unit, :-2 * w_shift_unit]

    shift_tensor = np.concatenate([top_left, top, top_right,
                                   left, center, right,
                                   bottom_left, bottom, bottom_right], axis=0)
    return shift_tensor

def init_spixel_grid(img_height, img_width, batch_size):
    curr_img_height = int(np.floor(img_height))
    curr_img_width = int(np.floor(img_width))

    all_h_coords = np.arange(0, curr_img_height, 1)
    all_w_coords = np.arange(0, curr_img_width, 1)
    curr_pxl_coord = np.array(
        np.meshgrid(
            all_h_coords,
            all_w_coords,
            indexing='ij'))
    coord_tensor = np.concatenate(
        [curr_pxl_coord[1:2, :, :], curr_pxl_coord[:1, :, :]])
    all_XY_feat = (
        torch.from_numpy(
            np.tile(
                coord_tensor, (batch_size, 1, 1, 1)).astype(
                np.float32)).cuda())

    return all_XY_feat

def build_LABXY_feat(label_in, XY_feat):
    img_lab = label_in.clone().type(torch.float)
    b, _, curr_img_height, curr_img_width = XY_feat.shape
    scale_img = F.interpolate(
        img_lab,
        size=(
            curr_img_height,
            curr_img_width),
        mode='nearest')
    LABXY_feat = torch.cat([scale_img, XY_feat], dim=1)

    return LABXY_feat

def poolfeat(input, prob, sp_h=2, sp_w=2):
    def feat_prob_sum(feat_sum, prob_sum, shift_feat):
        feat_sum += shift_feat[:, :-1, :, :]
        prob_sum += shift_feat[:, -1:, :, :]
        return feat_sum, prob_sum

    b, _, h, w = input.shape
    h_shift_unit = 1
    w_shift_unit = 1
    p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)
    feat_ = torch.cat([input, torch.ones([b, 1, h, w]).cuda()], dim=1)
    prob_feat = F.avg_pool2d(
        feat_ *
        prob.narrow(
            1,
            0,
            1),
        kernel_size=(
            sp_h,
            sp_w),
        stride=(
            sp_h,
            sp_w))
    temp = F.pad(prob_feat, p2d, mode='constant', value=0)
    send_to_top_left = temp[:, :, 2 * h_shift_unit:, 2 * w_shift_unit:]
    feat_sum = send_to_top_left[:, :-1, :, :].clone()
    prob_sum = send_to_top_left[:, -1:, :, :].clone()

    prob_feat = F.avg_pool2d(
        feat_ *
        prob.narrow(
            1,
            1,
            1),
        kernel_size=(
            sp_h,
            sp_w),
        stride=(
            sp_h,
            sp_w))
    top = F.pad(prob_feat, p2d, mode='constant', value=0)[
          :, :, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top)

    prob_feat = F.avg_pool2d(
        feat_ *
        prob.narrow(
            1,
            2,
            1),
        kernel_size=(
            sp_h,
            sp_w),
        stride=(
            sp_h,
            sp_w))
    top_right = F.pad(prob_feat, p2d, mode='constant', value=0)[
                :, :, 2 * h_shift_unit:, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)

    prob_feat = F.avg_pool2d(
        feat_ *
        prob.narrow(
            1,
            3,
            1),
        kernel_size=(
            sp_h,
            sp_w),
        stride=(
            sp_h,
            sp_w))
    left = F.pad(prob_feat, p2d, mode='constant', value=0)[
           :, :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)

    prob_feat = F.avg_pool2d(
        feat_ *
        prob.narrow(
            1,
            4,
            1),
        kernel_size=(
            sp_h,
            sp_w),
        stride=(
            sp_h,
            sp_w))
    center = F.pad(prob_feat, p2d, mode='constant', value=0)[
              :, :, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)

    prob_feat = F.avg_pool2d(
        feat_ *
        prob.narrow(
            1,
            5,
            1),
        kernel_size=(
            sp_h,
            sp_w),
        stride=(
            sp_h,
            sp_w))
    right = F.pad(prob_feat, p2d, mode='constant', value=0)[
            :, :, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)

    prob_feat = F.avg_pool2d(
        feat_ *
        prob.narrow(
            1,
            6,
            1),
        kernel_size=(
            sp_h,
            sp_w),
        stride=(
            sp_h,
            sp_w))
    bottom_left = F.pad(prob_feat, p2d, mode='constant', value=0)[
                  :, :, :-2 * h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)

    prob_feat = F.avg_pool2d(
        feat_ *
        prob.narrow(
            1,
            7,
            1),
        kernel_size=(
            sp_h,
            sp_w),
        stride=(
            sp_h,
            sp_w))
    bottom = F.pad(prob_feat, p2d, mode='constant', value=0)[
              :, :, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)

    prob_feat = F.avg_pool2d(
        feat_ *
        prob.narrow(
            1,
            8,
            1),
        kernel_size=(
            sp_h,
            sp_w),
        stride=(
            sp_h,
            sp_w))
    bottom_right = F.pad(prob_feat, p2d, mode='constant', value=0)[
                   :, :, :-2 * h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)

    pooled_feat = feat_sum / (prob_sum + 1e-8)

    return pooled_feat

def upfeat(input, prob, up_h=2, up_w=2):
    b, c, h, w = input.shape
    h_shift = 1
    w_shift = 1
    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = F.pad(input, p2d, mode='constant', value=0)

    gt_frm_top_left = F.interpolate(
        feat_pd[:, :, :-2 * h_shift, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum = gt_frm_top_left * prob.narrow(1, 0, 1)

    top = F.interpolate(feat_pd[:,
                        :,
                        :-2 * h_shift,
                        w_shift:-w_shift],
                        size=(h * up_h,
                              w * up_w),
                        mode='nearest')
    feat_sum += top * prob.narrow(1, 1, 1)

    top_right = F.interpolate(feat_pd[:,
                              :,
                              :-2 * h_shift,
                              2 * w_shift:],
                              size=(h * up_h,
                                    w * up_w),
                              mode='nearest')
    feat_sum += top_right * prob.narrow(1, 2, 1)

    left = F.interpolate(feat_pd[:,
                         :,
                         h_shift:-w_shift,
                         :-2 * w_shift],
                         size=(h * up_h,
                               w * up_w),
                         mode='nearest')
    feat_sum += left * prob.narrow(1, 3, 1)

    center = F.interpolate(input, (h * up_h, w * up_w), mode='nearest')
    feat_sum += center * prob.narrow(1, 4, 1)

    right = F.interpolate(feat_pd[:,
                          :,
                          h_shift:-w_shift,
                          2 * w_shift:],
                          size=(h * up_h,
                                w * up_w),
                          mode='nearest')
    feat_sum += right * prob.narrow(1, 5, 1)

    bottom_left = F.interpolate(feat_pd[:,
                                :,
                                2 * h_shift:,
                                :-2 * w_shift],
                                size=(h * up_h,
                                      w * up_w),
                                mode='nearest')
    feat_sum += bottom_left * prob.narrow(1, 6, 1)

    bottom = F.interpolate(feat_pd[:,
                           :,
                           2 * h_shift:,
                           w_shift:-w_shift],
                           size=(h * up_h,
                                 w * up_w),
                           mode='nearest')
    feat_sum += bottom * prob.narrow(1, 7, 1)

    bottom_right = F.interpolate(
        feat_pd[:, :, 2 * h_shift:, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_right * prob.narrow(1, 8, 1)

    return feat_sum

def compute_semantic_pos_loss(
        prob_in,
        labxy_feat,
        pos_weight=0.003,
        kernel_size=16):
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)

    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    loss_map = reconstr_feat[:, -2:, :, :] - labxy_feat[:, -2:, :, :]

    logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
    loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b
    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S

    loss_sum = 0.05 * (loss_sem + loss_pos)
    loss_sem_sum = 0.05 * loss_sem
    loss_pos_sum = 0.05 * loss_pos

    return loss_sum, loss_sem_sum, loss_pos_sum

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader,
                epochs, model_name, state_save_path, state_load_path=None):
    os.makedirs(state_save_path, exist_ok=True)
    if state_load_path is not None:
        model.load_state_dict(torch.load(state_load_path))

    num_step = len(train_loader)
    best_val_dice = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        train_dice_score = 0

        xy_feat1 = init_spixel_grid(112, 112, batch_size)
        xy_feat2 = init_spixel_grid(56, 56, batch_size)
        xy_feat3 = init_spixel_grid(28, 28, batch_size)
        xy_feat4 = init_spixel_grid(14, 14, batch_size)

        with tqdm(total=num_step, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(train_loader):
                images, masks = batch[0], batch[1]
                batch_step = images.shape[0]
                images = images.to(device)
                masks = masks.to(device)

                if iteration == num_step - 1:
                    xy_feat1 = init_spixel_grid(112, 112, batch_step)
                    xy_feat2 = init_spixel_grid(56, 56, batch_step)
                    xy_feat3 = init_spixel_grid(28, 28, batch_step)
                    xy_feat4 = init_spixel_grid(14, 14, batch_step)

                masks1 = F.interpolate(masks, size=(112, 112), mode='nearest')
                masks2 = F.interpolate(masks, size=(56, 56), mode='nearest')
                masks3 = F.interpolate(masks, size=(28, 28), mode='nearest')
                masks4 = F.interpolate(masks, size=(14, 14), mode='nearest')

                LABXY_feat_tensor1 = build_LABXY_feat(masks1, xy_feat1)
                LABXY_feat_tensor2 = build_LABXY_feat(masks2, xy_feat2)
                LABXY_feat_tensor3 = build_LABXY_feat(masks3, xy_feat3)
                LABXY_feat_tensor4 = build_LABXY_feat(masks4, xy_feat4)

                masks_pred, Q_prob_collect = model(images)

                slic_loss1, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[0], LABXY_feat_tensor1, pos_weight=0.003, kernel_size=2)
                slic_loss2, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[1], LABXY_feat_tensor2, pos_weight=0.003, kernel_size=2)
                slic_loss3, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[2], LABXY_feat_tensor3, pos_weight=0.003, kernel_size=2)
                slic_loss4, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[3], LABXY_feat_tensor4, pos_weight=0.003, kernel_size=1)

                slice_loss = slic_loss1 + slic_loss2 + slic_loss3 + slic_loss4

                loss_value = criterion(masks_pred, masks) + dice_loss_binary(torch.sigmoid(masks_pred), masks)
                loss_sum = loss_value + slice_loss * 0.3
                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()
                epoch_loss += loss_sum.item()

                masks_pred = (masks_pred > 0).float()
                train_dice_score += dice_metric_binary(masks_pred, masks)

                pbar.set_postfix(**{'loss': epoch_loss / (iteration + 1),
                                    'dice': train_dice_score.item() / (iteration + 1)})
                pbar.update(1)

        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_images, val_masks = val_batch[0].to(device), val_batch[1].to(device)
                val_preds, _ = model(val_images)
                val_preds = nn.Sigmoid()(val_preds)
                val_preds = (val_preds > 0.5).float()
                val_dice += dice_metric_binary(val_preds, val_masks).item()
        avg_val_dice = val_dice / len(val_loader)

        print(f"Validation Dice: {avg_val_dice:.4f}")
        scheduler.step()

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(state_save_path, model_name + '_best.pth'))
            print(f"Model improved. Best validation Dice: {best_val_dice:.4f}, model saved.")

def compute_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.0, 0.0
    elif pred.sum() == 0 or gt.sum() == 0:
        acc = (pred == gt).mean()
        return 0.0, acc, 0.0, 0.0, 0.0, 100.0

    dice = dc(pred, gt)
    acc = (pred == gt).mean()
    jaccard = jc(pred, gt)
    sen = sensitivity(pred, gt)
    pre = precision(pred, gt)

    try:
        hd = hd95(pred, gt)
    except Exception:
        hd = 100.0

    return dice, acc, jaccard, sen, pre, hd

def predict_images_for_fold(net, dataloader, device, output_path, compare_path):
    net.eval()
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(compare_path, exist_ok=True)

    metrics_sum = {'dice': 0, 'acc': 0, 'jac': 0, 'sen': 0, 'pre': 0, 'hd': 0}
    total_images = 0

    with tqdm(total=len(dataloader), desc='Predict images:', postfix=dict, mininterval=0.3) as pbar:
        for batch in dataloader:
            image, mask_true, image_filenames = batch
            image = image.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred, _ = net(image)
                mask_pred = torch.sigmoid(mask_pred)
                mask_pred = (mask_pred > 0.5).float()

            for s in range(mask_pred.shape[0]):
                mask_pred_s = mask_pred[s].cpu().numpy().squeeze()
                mask_true_s = mask_true[s].cpu().numpy().squeeze()

                mask_pred_bin = (mask_pred_s > 0.5).astype(np.float32)
                mask_pred_to_save = (mask_pred_bin * 255).astype(np.uint8)

                filename_prefix = os.path.splitext(image_filenames[s])[0]
                save_path = os.path.join(output_path, f'{filename_prefix}.png')
                cv2.imwrite(save_path, mask_pred_to_save)

                dice, acc, jac, sen, pre, hd = compute_metrics(mask_pred_bin, mask_true_s)

                metrics_sum['dice'] += dice
                metrics_sum['acc'] += acc
                metrics_sum['jac'] += jac
                metrics_sum['sen'] += sen
                metrics_sum['pre'] += pre
                metrics_sum['hd'] += hd
                total_images += 1

                save_prediction(mask_pred_to_save, mask_true_s, os.path.join(compare_path, f'{filename_prefix}.png'))

            pbar.set_postfix({'dice': f'{metrics_sum["dice"] / total_images:.4f}'})
            pbar.update(1)

    fold_metrics = {k: v / total_images for k, v in metrics_sum.items()}
    return fold_metrics

from matplotlib import pyplot as plt

def save_prediction(Prediction, Target, output_path, gap_width=0.01):
    Prediction = (Prediction * 255).astype(np.uint8)
    if isinstance(Target, torch.Tensor):
        Target = (Target * 255).cpu().squeeze().numpy().astype(np.uint8)
    else:
        Target = (Target * 255).squeeze().astype(np.uint8)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(wspace=gap_width)
    axs[0].imshow(Prediction, cmap='gray')
    axs[0].set_title('Prediction')
    axs[0].axis('off')
    axs[1].imshow(Target, cmap='gray')
    axs[1].set_title('Target')
    axs[1].axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

if __name__ == '__main__':
    from datasets.breast import BUSI_Dataset, STU_Dataset
    from sklearn.model_selection import KFold
    from torch.utils.data import Subset
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    batch_size = 16
    learning_rate = 2e-4
    total_epoch = 120
    k_folds = 5

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.Resize(224, 224, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(224, 224, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    mix_dataset = BUSI_Dataset(root_dir='/autodl-fs/data/code/Dataset_BUSI_with_GT', category='mix', transform=None)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(mix_dataset)):
        print(f"\n====================== Starting Fold {fold + 1}/{k_folds} ======================")

        model = Stage_SSM(num_class=1).to(device)
        model.load_backbone('/autodl-fs/data/segformer_b2_backbone_weights.pth')

        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-6,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=40, T_mult=2, eta_min=1e-6, last_epoch=-1
        )
        criterion = nn.BCEWithLogitsLoss()

        train_dataset = Subset(
            BUSI_Dataset(root_dir='/autodl-fs/data/code/Dataset_BUSI_with_GT', category='mix',
                         transform=train_transform),
            train_idx
        )
        test_dataset = Subset(
            BUSI_Dataset(root_dir='/autodl-fs/data/code/Dataset_BUSI_with_GT', category='mix', transform=val_transform),
            val_idx
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        model_save_path = f'./model_mix/fold_{fold + 1}'
        output_images_path = f'result/fold_{fold + 1}'
        compare_images_path = f'results_compare/fold_{fold + 1}'

        model_name = f'ssm_fold_{fold + 1}'
        train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=total_epoch,
            model_name=model_name,
            state_save_path=model_save_path
        )

        best_model_path = os.path.join(model_save_path, f'{model_name}_best.pth')
        model.load_state_dict(torch.load(best_model_path))

        print(f"\nEvaluating Fold {fold + 1}...")
        metrics = predict_images_for_fold(
            net=model,
            dataloader=test_loader,
            device=device,
            output_path=output_images_path,
            compare_path=compare_images_path
        )
        all_fold_results.append(metrics)

    print("\n\n" + "=" * 70)
    print("Fold       | Dice(%)   | Acc(%)   | Jac(%)   | Sen(%)   | Pre(%)   | HD")
    print("-" * 70)

    all_metrics = {'dice': [], 'acc': [], 'jac': [], 'sen': [], 'pre': [], 'hd': []}

    for i, res in enumerate(all_fold_results):
        print(
            f"Fold {i + 1:<5} | {res['dice'] * 100:<9.2f} | {res['acc'] * 100:<8.2f} | {res['jac'] * 100:<8.2f} | {res['sen'] * 100:<8.2f} | {res['pre'] * 100:<8.2f} | {res['hd']:<8.2f}")

        for k in all_metrics.keys():
            all_metrics[k].append(res[k])

    print("-" * 70)

    avg_metrics = {}
    std_metrics = {}

    for k in all_metrics.keys():
        values = np.array(all_metrics[k])
        avg_metrics[k] = values.mean()
        std_metrics[k] = values.std()

    print(
        f"Average    | {avg_metrics['dice'] * 100:<9.2f}±{std_metrics['dice'] * 100:<.2f} | "
        f"{avg_metrics['acc'] * 100:<8.2f}±{std_metrics['acc'] * 100:<.2f} | "
        f"{avg_metrics['jac'] * 100:<8.2f}±{std_metrics['jac'] * 100:<.2f} | "
        f"{avg_metrics['sen'] * 100:<8.2f}±{std_metrics['sen'] * 100:<.2f} | "
        f"{avg_metrics['pre'] * 100:<8.2f}±{std_metrics['pre'] * 100:<.2f} | "
        f"{avg_metrics['hd']:<8.2f}±{std_metrics['hd']:<.2f}")
    print("=" * 70)
