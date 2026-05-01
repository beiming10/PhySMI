# load_stc_dataset.py (修正版)

import os
import glob
from torch.utils.data import Dataset
from tifffile import imread
import numpy as np
import torchvision.transforms as transforms

class StcDataset(Dataset):
    def __init__(self, data_root, crop_size=128, scalefactor=250.0):
        """
        一个简单的数据加载器，用于无监督的StC图像库。
        Args:
            data_root (str): 存放单通道StC图像的文件夹路径。
            crop_size (int): 随机裁剪的图像大小。
            scalefactor (float): 归一化因子。
        """
        self.files = glob.glob(os.path.join(data_root, '*.tif')) + glob.glob(os.path.join(data_root, '*.tiff'))
        self.scalefactor = scalefactor
        
        # 定义变换流程
        # ToTensor() 会自动将 (H, W) 的Numpy数组或PIL Image转为 (1, H, W) 的Tensor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

    def __getitem__(self, index):
        # 循环读取，防止数据量少于训练迭代次数时出错
        img_path = self.files[index % len(self.files)]
        
        # 读取TIFF图像，确保为 (H, W) 的2D数组
        img = imread(img_path).astype(np.float32)
        
        # 确保图像是单通道的
        if img.ndim != 2:
            print(f"Warning: Image {os.path.basename(img_path)} has {img.ndim} dimensions, expected 2. Taking the first slice.")
            img = img[0]

        # --- **关键修正点** ---
        # 1. 不再手动增加维度。直接将 (H, W) 的Numpy数组传入
        # 2. 归一化被 ToTensor() 自动处理（因为它会将值范围缩放到[0,1]）
        #    但由于我们的scalefactor不同，我们先进行手动缩放。
        
        img = img / self.scalefactor
        
        # ToTensor 会将 (H,W) Numpy -> (1,H,W) Tensor, 并将值范围从 [0, 1] 保持不变 (因为是float32)
        # 这样 RandomCrop 接收到的就是正确形状的 (1, 256, 256) Tensor
        img_tensor = self.transform(img)
        
        return img_tensor

    def __len__(self):
        # 返回一个较大的数，确保迭代器不会过早耗尽，适用于无监督学习
        return len(self.files) * 100