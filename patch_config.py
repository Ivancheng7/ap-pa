from torch import optim
from darknet import *
from load_data import MaxProbExtractor_yolov2, MaxProbExtractor_yolov5, MeanProbExtractor_yolov5, \
    MeanProbExtractor_yolov2
from mmdet.apis.inference import InferenceDetector
from models.common import DetectMultiBackend
from models_yolov3.common import DetectMultiBackend_yolov3
from utils_yolov5.torch_utils import select_device, time_sync
import os
from mmdet.apis import (async_inference_detector, InferenceDetector,
                        init_detector, show_result_pyplot)


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        # self.img_dir = "datasets/RSOD-Dataset/aircraft/Annotation/JPEGImages"
        self.img_dir = "testing/plane_random_loc/clean"
        # self.lab_dir = "datasets/RSOD-Dataset/aircraft/Annotation/JPEGImages/labels-yolo"
        self.lab_dir = "testing/yolov5l_center_150_1024_yolov5l/clean/labels-yolo"
        self.printfile = "non_printability/30values.txt"
        self.patch_size = 50

        self.start_learning_rate = 0.03

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0

        self.batch_size = 1

        self.loss_target = lambda obj, cls: obj * cls  # self.loss_target(obj, cls) return obj * cls



class faster_rcnn(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
            """
            初始化对象检测类实例。
            设置模型配置文件、检查点文件、设备类型等参数，并初始化模型。
            """
            # 调用超类的初始化方法
            super().__init__()

            # 初始化模型特定的参数
            self.patch_name = 'ObjectOnlyPaper'
            self.max_tv = 0.165

            # 定义损失目标函数，这里简单地将对象作为损失目标
            self.loss_target = lambda obj, cls: obj

            # 配置文件路径，用于定义模型结构和训练参数
            self.config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
            # 模型检查点文件路径，用于加载预训练模型权重
            # 2080
            # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/faster-rcnn/epoch_12.pth'
            # 3080
            # self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/faster-rcnn/epoch_12.pth'
            self.checkpoint_file = 'chx/model-fasterrcnn.pth'

            # 设备类型，这里使用CUDA加速计算
            self.device = 'cuda:0'

            # 图像处理的大小配置
            self.img_size = 1024
            self.imgsz = (1024, 1024)
            # 设置检测结果的置信度和IOU阈值
            self.conf_thres = 0.4  # confidence threshold
            self.iou_thres = 0.45  # NMS IOU threshold
            # 最大检测数量限制
            self.max_det = 1000  # maximum detections per image
            # 类别过滤设置，None表示检测所有类别
            self.classes = None  # filter by class: --class 0, or --class 0 2 3
            # 是否使用类无差别的非最大抑制
            self.agnostic_nms = False  # class-agnostic NMS

            # 初始化检测模型
            self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)

            # 初始化推理检测器实例
            self.InferenceDetector = InferenceDetector()




patch_configs = {
    "base": BaseConfig,
    "faster-rcnn": faster_rcnn,
}
