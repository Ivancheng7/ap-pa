import cv2
import torch
from torchvision import transforms
import os
from PIL import Image
import random

if __name__ == "__main__":
    test_path = "images_p_grid_3x3_1.0"#补丁图像文件夹
    save_path = "trans_img" #变换后图像文件夹
    random_rotation = transforms.RandomRotation(degrees = 180)
    random_bright = transforms.ColorJitter(brightness=0.3,contrast=0.05, saturation=0.05)
    for i, img_path in enumerate(os.listdir(test_path)):
        image_path = os.path.join(test_path, img_path)
        imgc = Image.open(image_path)
        for i in range(1,6):
            img = random_bright(imgc)
            img = random_rotation(img)
            img.save(save_path++"/{}_{}.png".format(img_path.split("/")[-1].split(".")[0],i))