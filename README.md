# AP-PA (Adaptive-Patch-based Physical Attack)

## Introduction

In this paper, a novel adaptive-patch-based physical attack (AP-PA) framework is proposed, which aims to generate adversarial patches that are adaptive in both physical dynamics and varying scales, and by which the particular targets can be hidden from being detected. Furthermore, the adversarial patch is also gifted with attack effectiveness against all targets of the same class with a patch outside the target (No need to smear targeted objects) and robust enough in the physical world. In addition, a new loss is devised to consider more available information of detected objects to optimize the adversarial patch, which can significantly improve the patch's attack efficacy (Average precision drop up to $87.86\%$ and $85.48\%$ in white-box and black-box settings, respectively) and optimizing efficiency. We also establish one of the first comprehensive, coherent, and rigorous benchmarks to evaluate the attack efficacy of adversarial patches on aerial detection tasks. We summarize our algorithm in [Benchmarking Adversarial Patch Against Aerial Detection](https://ieeexplore.ieee.org/document/9965436)。

## Requirements:

* Pytorch 1.10

* Python 3.6

## Citation

If you use AP-PA method for attacks in your research, please consider citing

```
@article{lian2022benchmarking,
  title={Benchmarking Adversarial Patch Against Aerial Detection},
  author={Lian, Jiawei and Mei, Shaohui and Zhang, Shun and Ma, Mingyang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--16},
  year={2022},
  publisher={IEEE}
}
```
环境配置如下：
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

数据集为yolo格式，lables为txt格式
权重文件放在chx文件夹中
