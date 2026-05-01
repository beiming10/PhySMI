# load_mata_hsi_dataset.py (PIANet-D 适配版)

from torch.utils.data import Dataset
import os 
import glob
from PIL import Image
import numpy as np
import h5py
import random
from scipy.io import loadmat
try:
    from tifffile import imread, TiffFile
    TIFF_AVAILABLE = True
except ImportError:
    TIFF_AVAILABLE = False
    print("Warning: tifffile not available. TIFF support disabled.")

class TrainDataset(Dataset):
    # ## 修改 ##: __init__ 签名，接收 input_indices_a 和 input_indices_b
    def __init__(self, data_root, crop_size, aug=True,
                 use_tiff=True, hyper_frames=8, 
                 input_indices_a=None, input_indices_b=None, 
                 step=4, stride=64, scalefactor=250.0):
        
        self.crop_size = crop_size
        self.aug = aug
        self.stride = stride
        self.use_tiff = use_tiff
        self.hyper_frames = hyper_frames
        self.step = step
        self.scalefactor = scalefactor
        
        # ## 修改 ##: 存储两个索引列表
        # 验证并存储通道索引
        if not input_indices_a or not input_indices_b:
            raise ValueError("For PIANet-D, both 'input_indices_a' and 'input_indices_b' must be provided.")
        
        # 验证索引是有效的列表
        if not isinstance(input_indices_a, list) or not isinstance(input_indices_b, list):
            raise TypeError("'input_indices_a' and 'input_indices_b' must be lists of integers.")
        
        # 确保索引中没有重复值，且两个列表没有重叠
        combined_indices = input_indices_a + input_indices_b
        if len(combined_indices) != len(set(combined_indices)):
            print("'input_indices_a' and 'input_indices_b' should not have overlapping or duplicate indices.")
        
        self.input_indices_a = input_indices_a
        self.input_indices_b = input_indices_b
        
        print(f"TrainDataset: Using channel subset A: {self.input_indices_a}")
        print(f"TrainDataset: Using channel subset B: {self.input_indices_b}")

        # ## 修改 ##: 不再存储图像，而是存储“样本信息”以实现懒加载
        self.samples_info = []

        if use_tiff:
            self._prepare_tiff_samples(data_root)
        else:
            # self._prepare_mat_samples(data_root) # MAT文件也应采用类似逻辑
            raise NotImplementedError("MAT file lazy loading not implemented in this version.")

        # 基于 patch 数量计算总长度
        if not self.samples_info:
            self.length = 0
        else:
            h, w = self.samples_info[0]['shape']
            patch_per_line = (w - crop_size) // stride + 1 if w > crop_size else 1
            patch_per_colum = (h - crop_size) // stride + 1 if h > crop_size else 1
            self.patch_per_img = patch_per_line * patch_per_colum
            self.img_num = len(self.samples_info)
            self.length = self.patch_per_img * self.img_num
        
        print(f'Total training patches: {self.length}')
    
    def _prepare_tiff_samples(self, data_root):
        """只记录文件路径和样本信息，不加载图像内容"""
        if not TIFF_AVAILABLE:
            raise ImportError("tifffile package is required for TIFF support")
        
        tiff_data_path = os.path.join(data_root, 'Train/')
        tiff_list = sorted([f for f in glob.glob(os.path.join(tiff_data_path, '*')) if f.lower().endswith(('.tif', '.tiff'))])
        
        print(f'Preparing sample info from {len(tiff_list)} TIFF stack files in {tiff_data_path}')
        
        for tiff_path in tiff_list:
            try:
                # 使用 TiffFile 直接读取元数据以获取形状信息，避免加载整个文件
                with TiffFile(tiff_path) as tiff:
                    # 获取第一个系列的形状信息
                    if tiff.series and len(tiff.series) > 0:
                        # 对于3D堆栈，形状通常是(frames, height, width)
                        stack_shape = tiff.series[0].shape
                    else:
                        # 备选方案：使用第一页获取空间尺寸
                        page = tiff.pages[0]
                        # 对于单通道图像，添加一个通道维度
                        stack_shape = (len(tiff.pages), page.shape[0], page.shape[1])
            except Exception as e:
                print(f"Warning: Could not read shape from {os.path.basename(tiff_path)}. Skipping file. Error: {e}")
                continue
            
            if stack_shape[0] < self.hyper_frames:
                continue

            h, w = stack_shape[1], stack_shape[2]
            
            total_frames = stack_shape[0]
            for start_frame in range(0, total_frames - self.hyper_frames + 1, self.step):
                self.samples_info.append({
                    'path': tiff_path,
                    'start_frame': start_frame,
                    'shape': (h, w)
                })

    def augment(self, *imgs):
        """对传入的所有图像应用相同的随机变换"""
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        
        augmented_imgs = []
        for img in imgs:
            augmented_img = img.copy()
            for _ in range(rotTimes): augmented_img = np.rot90(augmented_img, axes=(1, 2))
            if vFlip: augmented_img = augmented_img[:, :, ::-1]
            if hFlip: augmented_img = augmented_img[:, ::-1, :]
            augmented_imgs.append(augmented_img)
            
        return augmented_imgs

    def __getitem__(self, idx):
        if self.length == 0:
            raise IndexError("Dataset is empty.")
            
        # 1. 计算当前idx对应哪个文件和哪个patch
        patch_per_line = (self.samples_info[0]['shape'][1] - self.crop_size) // self.stride + 1
        self.patch_per_img = patch_per_line * ((self.samples_info[0]['shape'][0] - self.crop_size) // self.stride + 1)
        
        img_idx = idx // self.patch_per_img
        patch_idx = idx % self.patch_per_img
        
        info = self.samples_info[img_idx]
        tiff_path = info['path']
        start_frame = info['start_frame']
        h, w = info['shape']
        
        # 2. 只在需要时从硬盘读取这一个文件
        tiff_stack = imread(tiff_path).astype(np.float32)
        
        # 3. 从完整的stack中切片出需要的数据块并归一化
        full_block = tiff_stack[start_frame : start_frame + self.hyper_frames] / self.scalefactor
        
        # 4. ## 修改 ##: 根据索引分裂成三个部分
        target_data = full_block
        subset_a_data = full_block[self.input_indices_a]
        subset_b_data = full_block[self.input_indices_b]
        
        # 5. 从数据块中裁剪出当前需要的patch
        h_idx = patch_idx // patch_per_line
        w_idx = patch_idx % patch_per_line
        h_start, w_start = h_idx * self.stride, w_idx * self.stride
        
        target_patch = target_data[:, h_start : h_start+self.crop_size, w_start : w_start+self.crop_size]
        subset_a_patch = subset_a_data[:, h_start : h_start+self.crop_size, w_start : w_start+self.crop_size]
        subset_b_patch = subset_b_data[:, h_start : h_start+self.crop_size, w_start : w_start+self.crop_size]

        # 6. 数据增强
        if self.aug:
            target_patch, subset_a_patch, subset_b_patch = self.augment(target_patch, subset_a_patch, subset_b_patch)
            
        return np.ascontiguousarray(target_patch), np.ascontiguousarray(subset_a_patch), np.ascontiguousarray(subset_b_patch)

    def __len__(self):
        return self.length

class ValidDataset(Dataset):
    # ## 修改 ##: __init__ 签名，以适配 PIANet-D
    def __init__(self, data_root, use_tiff=True, hyper_frames=8, 
                 input_indices_a=None, input_indices_b=None, 
                 step=4, scalefactor=250.0):
        self.hypers = []
        self.lcs_a = []
        self.lcs_b = []
        self.use_tiff = use_tiff
        self.hyper_frames = hyper_frames
        # 验证并存储通道索引
        if not input_indices_a or not input_indices_b:
            raise ValueError("For PIANet-D, both 'input_indices_a' and 'input_indices_b' must be provided.")
        
        # 验证索引是有效的列表
        if not isinstance(input_indices_a, list) or not isinstance(input_indices_b, list):
            raise TypeError("'input_indices_a' and 'input_indices_b' must be lists of integers.")
        
        # 确保索引中没有重复值，且两个列表没有重叠
        combined_indices = input_indices_a + input_indices_b
        if len(combined_indices) != len(set(combined_indices)):
            print("'input_indices_a' and 'input_indices_b' should not have overlapping or duplicate indices.")
        
        self.input_indices_a = input_indices_a
        self.input_indices_b = input_indices_b
        self.step = step
        self.scalefactor = scalefactor
        
        if use_tiff:
            self._load_tiff_data(data_root)
        else:
            raise NotImplementedError("MAT file loading not adapted for PIANet-D.")

    def _load_tiff_data(self, data_root):
        """验证集通常较小，可以直接加载到内存"""
        if not TIFF_AVAILABLE:
            raise ImportError("tifffile package is required for TIFF support")
        
        tiff_data_path = os.path.join(data_root, 'Valid/')
        tiff_list = sorted([f for f in glob.glob(os.path.join(tiff_data_path, '*')) if f.lower().endswith(('.tif', '.tiff'))])
        
        for tiff_path in tiff_list:
            try:
                tiff_stack = imread(tiff_path).astype(np.float32)
                total_frames = tiff_stack.shape[0]
                
                if total_frames < self.hyper_frames:
                    print(f"Warning: TIFF file {os.path.basename(tiff_path)} has {total_frames} frames, which is less than required {self.hyper_frames}. Skipping.")
                    continue
                
                for start_frame in range(0, total_frames - self.hyper_frames + 1, self.step):
                    full_block = tiff_stack[start_frame : start_frame + self.hyper_frames] / self.scalefactor
                    
                    self.hypers.append(full_block)
                    self.lcs_a.append(full_block[self.input_indices_a])
                    self.lcs_b.append(full_block[self.input_indices_b])
            except Exception as e:
                print(f"Warning: Could not read or process {os.path.basename(tiff_path)}. Skipping file. Error: {e}")
                continue

    def __getitem__(self, idx):
        # ## 修改 ##: 返回三元组
        return np.ascontiguousarray(self.hypers[idx]), \
               np.ascontiguousarray(self.lcs_a[idx]), \
               np.ascontiguousarray(self.lcs_b[idx])
    
    def __len__(self):
        return len(self.hypers)