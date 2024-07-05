
import os
import time
import warnings

import pandas as pd
import torch
import wandb
from tensorboardX import SummaryWriter
from torch import autograd
from torchvision import transforms
from tqdm import tqdm

import patch_config
from load_data import *

warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = '7'

# Expand to show
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)


class PatchTrainer(object):
    def __init__(self, mode):
            """
            初始化检测器。

            根据给定的模式(mode)配置和初始化各种组件，包括模型、推理检测器、补丁应用器、
            补丁转换器、NPS计算器和总变差计算器。

            参数:
            mode: 模式字符串，用于选择特定的配置。
            """
            # 存储模式
            self.mode = mode
            # 初始化epoch长度，默认为0
            self.epoch_length = 0
            # 根据模式加载配置
            self.config = patch_config.patch_configs[mode]()

            # 将模型设置为评估模式并转移到CUDA设备
            self.model = self.config.model.eval().cuda()
            # 将推理检测器转移到CUDA设备
            self.InferenceDetector = self.config.InferenceDetector.cuda()
            # 注释掉的概率提取器初始化代码
            # self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()

            # 初始化补丁应用器并转移到CUDA设备
            self.patch_applier = PatchApplier().cuda()
            # 初始化补丁转换器并转移到CUDA设备
            self.patch_transformer = PatchTransformer().cuda()

            # 初始化NPS计算器并转移到CUDA设备
            self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
            # 初始化总变差计算器并转移到CUDA设备
            self.total_variation = TotalVariation().cuda()

    def train(self):
        """
        训练对抗样本生成器。
        该函数初始化对抗补丁并将其应用于图像批处理数据集，通过优化对抗补丁来降低检测器的性能。
    
        :return: 无
        """
    
        # 初始化参数
        img_size = 1024  # 图像大小
        batch_size = self.config.batch_size  # 批处理大小
        n_epochs = 1000  # 训练轮数
    
        # 最大类别数，基于数据集的标注类别数量
        max_lab = 59 + 1  # YOLOv5l 的最大类别数
    
        # 时间戳，用于记录训练开始时间
        time_str = time.strftime("%Y%m%d-%H%M%S")
    
        # 生成初始的对抗补丁
        adv_patch_cpu = self.generate_patch("gray")  # 从灰度图像生成对抗补丁
        # adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")
        # 设置对抗补丁的梯度计算属性
        adv_patch_cpu.requires_grad_(True)
    
        # 构建数据加载器
        train_loader = torch.utils.data.DataLoader(
            InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size, shuffle=False),
            batch_size=batch_size,
            shuffle=True,
            num_workers=10)
    
        # 计算一个epoch的长度
        self.epoch_length = len(train_loader)
    
        # 初始化优化器和学习率调度器
        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
    
        # 使用Wandb进行实验跟踪
        wandb.init(project="Adversarial-attack")
        wandb.watch(self.model, log="all")
    
        # 记录训练开始时间
        et0 = time.time()
    
        # 开始训练循环
        for epoch in range(n_epochs):
            # 初始化 epoch 的损失值
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
        
            # 记录批次开始时间
            bt0 = time.time()
        
            # 遍历数据集
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}', total=self.epoch_length):
                # 在此开启梯度检测异常
                with autograd.detect_anomaly():
                    # 将图像和标签移动到GPU
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                
                    # 将对抗补丁移动到GPU
                    adv_patch = adv_patch_cpu.cuda()
                
                    # 应用对抗补丁到图像批次
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    to_pil_image = transforms.ToPILImage()

                    # 转换张量为PIL图像
                    pil_image = to_pil_image(tensor)

                    # 保存图像到文件
                    pil_image.save(f'image_{epoch}_{i_batch}.png')  # 保存图像
                    # 对图像进行插值缩放
                    p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))
                
                    # 将处理后的图像批次转换回CPU并进行预处理
                    p_img_batch_cpu = p_img_batch.clone()
                    p_img_batch_cpu = p_img_batch_cpu[0].detach().cpu().numpy()
                    p_img_batch_cpu = p_img_batch_cpu.reshape(1024, 1024, 3)
                    p_img_batch_cpu = p_img_batch_cpu * 255
                
                    # 进行检测器推理
                    data = self.InferenceDetector(self.model, p_img_batch_cpu)
                    data['img'][0] = p_img_batch
                    output = self.model(return_loss=False, rescale=True, **data)
                
                    if self.mode == 'faster-rcnn' or self.mode == 'ssd':
                        if len(output[0][0])==0:
                            # max_prob = torch.tensor(float(0))
                            mean_prob = torch.tensor(float(0))
                        else:
                            # max_prob = output[0][0][0][4]
                            mean_prob = torch.mean(output[0][0][:, 4])
                    elif self.mode == 'swin':
                        if len(output[0][0][0])==0:
                            # max_prob = torch.tensor(float(0))
                            mean_prob = torch.tensor(float(0))
                        else:
                            # max_prob = output[0][0][0][0][4]
                            mean_prob = torch.mean(output[0][0][0][:, 4])
                
                    # 计算NPS和TV损失
                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)
                
                    nps_loss = nps * 0.01
                    tv_loss = tv * 2.5
                
                    # 计算检测损失
                    det_loss = torch.mean(mean_prob)
                
                    # 总损失为检测损失、NPS损失和TV损失之和
                    loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                
                    # 更新损失值累计
                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss
                
                    # 反向传播和优化
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0, 1)  # 保持对抗补丁在图像范围内
                
                    # 每100个批次记录一次Wandb日志
                    if i_batch % 100 == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        wandb.log({
                            "Patches": wandb.Image(adv_patch_cpu, caption="patch{}".format(iteration)),
                            "tv_loss": tv_loss,
                            "nps_loss": nps_loss,
                            "det_loss": det_loss,
                            "total_loss": loss,
                        })
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    # 清理内存
                    del img_batch, lab_batch, adv_patch, adv_batch_t, p_img_batch_cpu, data, output, mean_prob, nps, tv, det_loss, p_img_batch, nps_loss, tv_loss, loss
                    torch.cuda.empty_cache()
                
                    # 更新批次开始时间
                    bt0 = time.time()

            et1 = time.time()
            ep_det_loss = ep_det_loss / len(train_loader)
            ep_nps_loss = ep_nps_loss / len(train_loader)
            ep_tv_loss = ep_tv_loss / len(train_loader)
            ep_loss = ep_loss / len(train_loader)
            # 更新学习率调度器
            scheduler.step(ep_loss)
        
            # 打印 epoch 摘要
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1 - et0)
            
            # 更新训练开始时间
            et0 = time.time()

    def generate_patch(self, type):

        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

    def read_image(self, path):
        """
        读取图片
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def main():
    trainer = PatchTrainer('faster-rcnn')
    trainer.train()


if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=0,1 python train_patch_mmdetection.py
# pip install wandb --upgrade
