import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch import optim
import cv2
import numpy as np

from utils.dataloader_alone import Allage_Dataset
from utils.post_process import post_process
from utils.metrics import dice_metric_binary, dice_metric_multiclass
from utils.loss import dice_loss_binary, dice_loss_multiclass

def predict_images(net, dataloader, device, output_path, out_image=False):
    net.eval()
    num_step = len(dataloader)

    with tqdm(total=num_step, desc='predict images:', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(dataloader):
            image, mask_true, alone_image_file = batch[0], batch[1], batch[2]
            image = image.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred = net(image)
                mask_pred = torch.sigmoid(mask_pred)
                mask_pred = torch.gt(mask_pred, 0.5)
                mask_pred = mask_pred.type(torch.float32)
                for s in range(mask_pred.shape[0]):
                    mask_pred_s = mask_pred[s]
                    mask_pred_s = mask_pred_s.cpu().numpy()
                    mask_pred_s = np.squeeze(mask_pred_s)
                    mask_pred_s = post_process(mask_pred_s)

                    alone_image_file_s = alone_image_file[s]
                    alone_image_file_s_prefix = os.path.splitext(alone_image_file_s)[0]

                    if out_image:
                        mask_pred_s = np.array(mask_pred_s, dtype=np.float32)
                        mask_pred_s = np.where(mask_pred_s == 1, 255, mask_pred_s)
                        cv2.imwrite(output_path + alone_image_file_s_prefix + '.png', mask_pred_s)
                    else:
                        np.save(output_path + alone_image_file_s_prefix + '.npy', mask_pred_s)

            pbar.set_postfix(**{'mask_pred': mask_pred.shape})
            pbar.update(1)

def predict_images_mc(net, dataloader, device, output_path, out_image=False, SSM=False, postprocess=False):
    net.eval()
    num_step = len(dataloader)

    with tqdm(total=num_step, desc='predict images:', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(dataloader):
            image, mask_true, alone_image_file = batch[0], batch[1], batch[2]
            image = image.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                if SSM:
                    mask_pred, _ = net(image)
                else:
                    mask_pred = net(image)

                mask_pred = F.softmax(mask_pred, dim=1)
                mask_pred = mask_pred.argmax(dim=1)
                mask_pred = mask_pred.type(torch.float32)

                for s in range(mask_pred.shape[0]):
                    mask_pred_s = mask_pred[s]
                    mask_pred_s = mask_pred_s.cpu().numpy()
                    mask_pred_s = np.squeeze(mask_pred_s)
                    if postprocess:
                        mask_pred_s = post_process(mask_pred_s)

                    alone_image_file_s = alone_image_file[s]
                    alone_image_file_s_prefix = os.path.splitext(alone_image_file_s)[0]

                    if out_image:
                        mask_pred_s = np.array(mask_pred_s, dtype=np.float32)
                        mask_pred_s = np.where(mask_pred_s == 1, 255, mask_pred_s)
                        cv2.imwrite(output_path + alone_image_file_s_prefix + '.png', mask_pred_s)
                    else:
                        np.save(output_path + alone_image_file_s_prefix + '.npy', mask_pred_s)

            pbar.set_postfix(**{'mask_pred': mask_pred.shape})
            pbar.update(1)

def predict_images_mc_adult(net, dataloader, device, output_path, out_image=False, SSM=False, postprocess=False):
    net.eval()
    num_step = len(dataloader)

    with tqdm(total=num_step, desc='predict images:', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(dataloader):
            image, mask_true, alone_image_file, patient_name = batch[0], batch[1], batch[2], batch[3]
            image = image.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                if SSM:
                    mask_pred, _ = net(image)
                else:
                    mask_pred = net(image)

                mask_pred = F.softmax(mask_pred, dim=1)
                mask_pred = mask_pred.argmax(dim=1)
                mask_pred = mask_pred.type(torch.float32)

                for s in range(mask_pred.shape[0]):
                    mask_pred_s = mask_pred[s]
                    mask_pred_s = mask_pred_s.cpu().numpy()
                    mask_pred_s = np.squeeze(mask_pred_s)
                    if postprocess:
                        mask_pred_s = post_process(mask_pred_s)

                    alone_image_file_s = alone_image_file[s]
                    alone_image_file_s_prefix = os.path.splitext(alone_image_file_s)[0]

                    patient_name_s = patient_name[s]
                    patient_path = os.path.join(output_path, patient_name_s) + '/'
                    if not os.path.exists(patient_path):
                        os.mkdir(patient_path)

                    if out_image:
                        mask_pred_s = np.array(mask_pred_s, dtype=np.float32)
                        mask_pred_s = np.where(mask_pred_s == 1, 255, mask_pred_s)
                        cv2.imwrite(patient_path + alone_image_file_s_prefix + '.png', mask_pred_s)
                    else:
                        np.save(patient_path + alone_image_file_s_prefix + '.npy', mask_pred_s)

            pbar.set_postfix(**{'mask_pred': mask_pred.shape})
            pbar.update(1)

def predict_images_2folder(net, sequence_image_npy, batch_size, new_shape, device, output_path, out_image=False):
    net.eval()
    num_step = len(os.listdir(sequence_image_npy))

    with tqdm(total=num_step, desc='predict images:', postfix=dict, mininterval=0.3) as pbar:
        for subfolder_name in os.listdir(sequence_image_npy):
            subfolder_path = os.path.join(sequence_image_npy, subfolder_name)
            sub_output_path = os.path.join(output_path, subfolder_name)
            if not os.path.exists(sub_output_path):
                os.mkdir(sub_output_path)

            subfolder_path_list = os.listdir(subfolder_path)
            num_subfolder_path_list = len(subfolder_path_list)
            subfolder_path_list = sorted(subfolder_path_list, key=lambda x: str(x.split('.')[0].split('-')[-1]))
            subfolder_path_list = sorted(subfolder_path_list, key=lambda x: len(x.split('.')[0].split('-')[-1]))

            alone_image_npy_list = []
            alone_image_file = []
            for index, subsubfile_name in enumerate(subfolder_path_list):
                index = index + 1
                subsubfile_path = os.path.join(subfolder_path, subsubfile_name)
                alone_image_npy = np.load(subsubfile_path)
                alone_image_npy = np.array(alone_image_npy, dtype=np.float32)
                alone_image_npy = alone_image_npy / 255
                alone_image_npy = alone_image_npy[..., np.newaxis]
                alone_image_npy = cv2.resize(alone_image_npy, new_shape)
                alone_image_npy = alone_image_npy[np.newaxis, ...]
                alone_image_npy = torch.from_numpy(alone_image_npy)

                alone_image_npy_list.append(alone_image_npy)
                alone_image_file.append(subsubfile_name)
                if index % batch_size == 0 or index == num_subfolder_path_list:
                    image = torch.stack(alone_image_npy_list)
                    image = image.to(device=device, dtype=torch.float32)

                    with torch.no_grad():
                        mask_pred = net(image)
                        mask_pred = torch.sigmoid(mask_pred)
                        mask_pred = torch.gt(mask_pred, 0.5)
                        mask_pred = mask_pred.type(torch.float32)
                        for s in range(mask_pred.shape[0]):
                            mask_pred_s = mask_pred[s]
                            mask_pred_s = mask_pred_s.cpu().numpy()
                            mask_pred_s = np.squeeze(mask_pred_s)
                            mask_pred_s = post_process(mask_pred_s)
                            alone_image_file_s = alone_image_file[s]
                            alone_image_file_s_prefix = os.path.splitext(alone_image_file_s)[0]
                            if out_image:
                                mask_pred_s = np.array(mask_pred_s, dtype=np.float32)
                                mask_pred_s = np.where(mask_pred_s == 1, 255, mask_pred_s)
                                cv2.imwrite(sub_output_path + '/' + alone_image_file_s_prefix + '.png', mask_pred_s)
                            else:
                                np.save(sub_output_path + '/' + alone_image_file_s_prefix + '.npy', mask_pred_s)
                    alone_image_npy_list = []
                    alone_image_file = []
            pbar.update(1)

def predict_images_2folder_mc(net, sequence_image_npy, batch_size, new_shape, device, output_path, out_image=False, post_process=False, SSM=False, nature=False):
    net.eval()
    num_step = len(os.listdir(sequence_image_npy))

    with tqdm(total=num_step, desc='predict images:', postfix=dict, mininterval=0.3) as pbar:
        for subfolder_name in os.listdir(sequence_image_npy):
            subfolder_path = os.path.join(sequence_image_npy, subfolder_name)
            sub_output_path = os.path.join(output_path, subfolder_name)
            if not os.path.exists(sub_output_path):
                os.mkdir(sub_output_path)

            subfolder_path_list = os.listdir(subfolder_path)
            num_subfolder_path_list = len(subfolder_path_list)
            subfolder_path_list = sorted(subfolder_path_list, key=lambda x: str(x.split('.')[0].split('-')[-1]))
            subfolder_path_list = sorted(subfolder_path_list, key=lambda x: len(x.split('.')[0].split('-')[-1]))

            alone_image_npy_list = []
            alone_image_file = []
            for index, subsubfile_name in enumerate(subfolder_path_list):
                index = index + 1
                subsubfile_path = os.path.join(subfolder_path, subsubfile_name)
                alone_image_npy = np.load(subsubfile_path)
                if nature:
                    alone_image_npy = alone_image_npy[:,:,0]
                alone_image_npy = np.array(alone_image_npy, dtype=np.float32)
                alone_image_npy = alone_image_npy / 255
                alone_image_npy = alone_image_npy[..., np.newaxis]
                alone_image_npy = cv2.resize(alone_image_npy, new_shape)
                alone_image_npy = alone_image_npy[np.newaxis, ...]
                alone_image_npy = torch.from_numpy(alone_image_npy)

                alone_image_npy_list.append(alone_image_npy)
                alone_image_file.append(subsubfile_name)
                if index % batch_size == 0 or index == num_subfolder_path_list:
                    image = torch.stack(alone_image_npy_list)
                    image = image.to(device=device, dtype=torch.float32)

                    with torch.no_grad():
                        if SSM:
                            mask_pred, _ = net(image)
                        else:
                            mask_pred = net(image)
                        mask_pred = F.softmax(mask_pred, dim=1)
                        mask_pred = mask_pred.argmax(dim=1)
                        mask_pred = mask_pred.type(torch.float32)
                        for s in range(mask_pred.shape[0]):
                            mask_pred_s = mask_pred[s]
                            mask_pred_s = mask_pred_s.cpu().numpy()
                            mask_pred_s = np.squeeze(mask_pred_s)
                            mask_pred_s = mask_pred_s[..., np.newaxis]
                            mask_pred_s = cv2.resize(mask_pred_s, new_shape)
                            if post_process:
                                mask_pred_s = cv2.medianBlur(np.uint8(mask_pred_s), 45)

                            alone_image_file_s = alone_image_file[s]
                            alone_image_file_s_prefix = os.path.splitext(alone_image_file_s)[0]
                            if out_image:
                                mask_pred_s = np.array(mask_pred_s, dtype=np.float32)
                                mask_pred_s = np.where(mask_pred_s == 1, 255, mask_pred_s)
                                mask_pred_s = np.where(mask_pred_s == 2, 125, mask_pred_s)
                                cv2.imwrite(sub_output_path + '/' + alone_image_file_s_prefix + '.png', mask_pred_s)
                            else:
                                np.save(sub_output_path + '/' + alone_image_file_s_prefix + '.npy', mask_pred_s)
                    alone_image_npy_list = []
                    alone_image_file = []
            pbar.update(1)

def predict_images_3folder_mc(net, sequence_image_npy, batch_size, new_shape, device, output_path, out_image=False, post_process=False, SSM=False, nature=False):
    for subfold_name in os.listdir(sequence_image_npy):
        subfold_path = os.path.join(sequence_image_npy, subfold_name)
        output_subfold_path = os.path.join(output_path, subfold_name)
        if not os.path.exists(output_subfold_path):
            os.mkdir(output_subfold_path)
        predict_images_2folder_mc(net, subfold_path, batch_size, new_shape, device, output_subfold_path, out_image=out_image, post_process=post_process, SSM=SSM, nature=nature)

def predict_npytoimg_2folder_mc(raw_path, npy_path, greyimg_path, colorimg_path, imgwithmask_path, shape=None, nature=False):
    for npy_subfold_name in os.listdir(npy_path):
        npy_subfold_path = os.path.join(npy_path, npy_subfold_name)
        greyimg_subfold_path = os.path.join(greyimg_path, npy_subfold_name)
        colorimg_subfold_path = os.path.join(colorimg_path, npy_subfold_name)
        imgwithmask_subfold_path = os.path.join(imgwithmask_path, npy_subfold_name)
        raw_subfold_path = os.path.join(raw_path, npy_subfold_name)

        if not os.path.exists(greyimg_subfold_path):
            os.mkdir(greyimg_subfold_path)
        if not os.path.exists(colorimg_subfold_path):
            os.mkdir(colorimg_subfold_path)
        if not os.path.exists(imgwithmask_subfold_path):
            os.mkdir(imgwithmask_subfold_path)

        for npy_subsubfile_name in os.listdir(npy_subfold_path):
            subsubfile_name_prefix = os.path.splitext(npy_subsubfile_name)[0]
            npy_subsubfile_path = os.path.join(npy_subfold_path, npy_subsubfile_name)
            npy_subsubfile_mat = np.load(npy_subsubfile_path)

            raw_subsubfile_path = os.path.join(raw_subfold_path, npy_subsubfile_name)
            raw_subsubfile_mat = np.load(raw_subsubfile_path)
            if nature:
                raw_subsubfile_mat = raw_subsubfile_mat[:,:,0]

            if shape is not None:
                raw_subsubfile_mat = raw_subsubfile_mat[..., np.newaxis]
                raw_subsubfile_mat = cv2.resize(raw_subsubfile_mat, shape, interpolation=cv2.INTER_NEAREST)

            img_subsubfile_mat = np.array(npy_subsubfile_mat, dtype=np.float32)
            img_subsubfile_mat = np.where(img_subsubfile_mat == 1, 255, img_subsubfile_mat)
            img_subsubfile_mat = np.where(img_subsubfile_mat == 2, 125, img_subsubfile_mat)
            cv2.imwrite(greyimg_subfold_path + '/' + subsubfile_name_prefix + '.png', img_subsubfile_mat)

            npy_subsubfile_mat = np.tile(npy_subsubfile_mat, (3, 1, 1))
            npy_subsubfile_mat = np.swapaxes(npy_subsubfile_mat, 0, 1)
            npy_subsubfile_mat = np.swapaxes(npy_subsubfile_mat, 1, 2)

            raw_subsubfile_mat = np.tile(raw_subsubfile_mat, (3, 1, 1))
            raw_subsubfile_mat = np.swapaxes(raw_subsubfile_mat, 0, 1)
            raw_subsubfile_mat = np.swapaxes(raw_subsubfile_mat, 1, 2)

            index = np.where(npy_subsubfile_mat == [1, 1, 1])
            for num0 in range(len(index[0])):
                npy_subsubfile_mat[index[0][num0], index[1][num0], :] = [255.0, 0, 0]
            index = np.where(npy_subsubfile_mat == [2, 2, 2])
            for num0 in range(len(index[0])):
                npy_subsubfile_mat[index[0][num0], index[1][num0], :] = [0, 255.0, 0]
            cv2.imwrite(colorimg_subfold_path + '/' + subsubfile_name_prefix + '.png', npy_subsubfile_mat)

            raw_subsubfile_mat = np.array(raw_subsubfile_mat, dtype=np.float64)
            npy_subsubfile_mat = np.array(npy_subsubfile_mat, dtype=np.float64)
            image_with_mask_mat = cv2.addWeighted(raw_subsubfile_mat, 1.0, npy_subsubfile_mat, 0.5, 1)
            cv2.imwrite(imgwithmask_subfold_path + '/' + subsubfile_name_prefix + '.png', image_with_mask_mat)

def predict_npytoimg_3folder_mc(raw_path, npy_path, greyimg_path, colorimg_path, imgwithmask_path, shape=None, nature=False):
    for npy_subfold_name in os.listdir(npy_path):
        npy_subfold_path = os.path.join(npy_path, npy_subfold_name)
        greyimg_subfold_path = os.path.join(greyimg_path, npy_subfold_name)
        colorimg_subfold_path = os.path.join(colorimg_path, npy_subfold_name)
        imgwithmask_subfold_path = os.path.join(imgwithmask_path, npy_subfold_name)
        raw_subfold_path = os.path.join(raw_path, npy_subfold_name)

        if not os.path.exists(greyimg_subfold_path):
            os.mkdir(greyimg_subfold_path)
        if not os.path.exists(colorimg_subfold_path):
            os.mkdir(colorimg_subfold_path)
        if not os.path.exists(imgwithmask_subfold_path):
            os.mkdir(imgwithmask_subfold_path)
        predict_npytoimg_2folder_mc(raw_subfold_path, npy_subfold_path, greyimg_subfold_path, colorimg_subfold_path, imgwithmask_subfold_path, shape=shape, nature=nature)
