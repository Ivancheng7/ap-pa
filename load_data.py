import fnmatch
import math
import os
import sys
import time
from operator import itemgetter

import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import get_region_boxes, nms

from darknet import Darknet

from median_pool import MedianPool2d
from utils_yolov5.general import non_max_suppression



class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),
                                               requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array + 0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]  # test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)



class PatchTransformer:
    def __init__(self):
        # 调用父类的初始化方法
        super(PatchTransformer, self).__init__()
        # 设置对比度的最小和最大值，用于调整图像的对比度
        self.min_contrast = 0.6
        self.max_contrast = 1.4
        
        # 设置亮度的最小和最大值，用于调整图像的亮度
        self.min_brightness = -0.2
        self.max_brightness = 0.2
        
            # 设置噪声因子，用于添加随机噪声到图像中
        self.noise_factor = 0.20
        
            # 设置旋转的角度范围，以弧度表示，用于随机旋转图像
        self.minangle = -60 / 180 * math.pi
        self.maxangle = 60 / 180 * math.pi
        
        # 初始化中值滤波器，用于去除噪声和保持边缘清晰
        self.medianpooler = MedianPool2d(7, same=True)
        
         
    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):
        """
        对给定的对抗补丁进行前向传播，应用一系列数据增强操作，包括旋转、缩放和随机位置调整。
        这些操作旨在增强模型的泛化能力，使其更难以被对抗样本欺骗。

        参数:
        - adv_patch: 输入的对抗补丁，需要应用数据增强的操作。
        - lab_batch: 标签批次，包含目标物体的位置和大小信息。
        - img_size: 目标图像的大小，用于调整补丁和标签的大小以适应目标图像。
        - do_rotate: 是否执行旋转数据增强，默认为True。
        - rand_loc: 是否在目标物体的位置上添加随机偏移，默认为True。

        返回:
        - 经过数据增强后的对抗补丁，大小和目标图像一致，并应用了旋转、缩放和随机位置调整。
        """
        # 使用中值池化处理对抗补丁
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        # 计算补丁大小与目标图像大小之间的差距，并为补丁添加合适的padding
        pad = (img_size - adv_patch.size(-1)) / 2

        # 扩展补丁批次大小，使其与标签批次大小匹配
        adv_patch = adv_patch.unsqueeze(0)
        adv_batch = adv_batch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)

        # 生成随机的对比度、亮度和噪声值，用于后续的数据增强
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

        # 应用对比度、亮度和噪声数据增强，并限制值的范围
        adv_batch = adv_batch * contrast + brightness + noise
        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # 根据标签生成掩码，用于在对抗样本中忽略非目标区域
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_mask = cls_ids.expand(-1, -1, 3).unsqueeze(-1).expand(-1, -1, -1, adv_batch.size(3)).unsqueeze(-1).expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask

        # 为补丁和掩码添加padding，使其与目标图像大小一致
        mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)

        # 如果启用旋转，生成随机旋转角度
        if do_rotate:
            angle = torch.cuda.FloatTensor(lab_batch.size(0) * lab_batch.size(1)).uniform_(self.minangle, self.maxangle)
        else:
            angle = torch.cuda.FloatTensor(lab_batch.size(0) * lab_batch.size(1)).fill_(0)

        # 根据标签信息计算目标物体的缩放和位置，并应用随机偏移
        lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        if rand_loc:
            off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
            target_x = target_x + off_x
            off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))
            target_y = target_y + off_y

        # 根据计算得到的目标位置和大小，进行旋转和缩放操作
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        # 应用掩码，得到最终的增强后的对抗样本
        adv_batch_t = adv_batch_t * msk_batch_t


class PatchApplier(nn.Module):

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch




class InriaDataset(Dataset):
    """
    用于加载自己的数据集
    属性：
    len: 数据集中元素的数量，以整数形式表示。
    img_dir: 包含数据集图像的目录。
    lab_dir: 包含数据集标签的目录。
    img_names: img_dir中所有图像文件名的列表。
    shuffle: 一个布尔值，表示是否打乱数据集的顺序。

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):  # check to see if label file contains data.
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)  # Make the labels of different images have same size. batch_size=1时不需要
        return image, label

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h / w)
        resize = transforms.Resize((self.imgsize, self.imgsize))
        padded_img = resize(padded_img)  # choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if (pad_size > 0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        return padded_lab

