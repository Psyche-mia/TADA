from PIL import Image
from glob import glob
import os
from torch.utils.data import Dataset
import math
import numpy as np
import random
from .imagenet_dic import IMAGENET_DIC

from PIL import Image
from glob import glob
import os
from torch.utils.data import Dataset
import math
import numpy as np
import random

def get_mvtec_dataset(data_root, config, class_num=None, random_crop=False, random_flip=False, split_ratio=0.8):
    """
    获取训练、测试和（可选）目标数据集。

    参数:
        data_root (str): 数据集根目录。
        config: 配置对象，包含类别和图像尺寸等信息。
        class_num: 类别编号（可选）。
        random_crop (bool): 是否随机裁剪。
        random_flip (bool): 是否随机翻转。
        split_ratio (float): 训练集和测试集划分比例。
        target_folder (str): 目标文件夹路径（可选）。

    返回:
        tuple: 包含训练集和测试集（始终返回），以及目标数据集（如果提供）。
    """
    category_name = config.data.category
    all_data_paths = MVTec_dataset.get_all_data_paths(data_root, category_name=category_name)
    random.shuffle(all_data_paths)
    split_index = int(len(all_data_paths) * split_ratio)
    
    train_paths = all_data_paths[:split_index]
    test_paths = all_data_paths[split_index:]
    
    # 创建训练集和测试集
    train_dataset = MVTec_dataset(train_paths, mode='train', img_size=config.data.image_size,
                                  random_crop=random_crop, random_flip=random_flip)
    test_dataset = MVTec_dataset(test_paths, mode='test', img_size=config.data.image_size,
                                 random_crop=random_crop, random_flip=random_flip)

    # 如果提供目标文件夹路径，则创建目标数据集
    target_folder = config.data.target_folder
    if target_folder:
        target_image_paths = MVTec_dataset.get_target_image_paths(data_root, category_name=category_name, target_folder=target_folder)
        target_dataset = MVTec_dataset(target_image_paths, mode='target', img_size=config.data.image_size,
                                       random_crop=random_crop, random_flip=random_flip)
        return train_dataset, test_dataset, target_dataset
    else:
        return train_dataset, test_dataset


###################################################################

class MVTec_dataset(Dataset):
    def __init__(self, image_paths, mode='train', img_size=512, random_crop=True, random_flip=False):
        super().__init__()
        self.image_paths = image_paths
        self.img_size = img_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.target_folder = target_folder  # New argument for target folder

        # if mode == 'target':
        #     self.image_paths = self._get_target_image_paths(target_folder)
        # else:
        #     self.target_image_paths = None

    @staticmethod
    def get_all_data_paths(image_root, category_name=None):
        """获取所有的图片路径"""
        if category_name is not None:
            data_dir = os.path.join(image_root, category_name, 'image', 'good', '*.png')
        else:
            data_dir = os.path.join(image_root, '*', 'image', 'good', '*.png')
        return sorted(glob(data_dir))

    def get_target_image_paths(self, category_name=None, target_folder=None):
        """根据target_folder获取目标文件夹中的图像路径"""
        data_dir = os.path.join(image_root, category_name, 'image', target_folder, '*.png')
        return sorted(glob(target_folder))  # Assuming target_folder is a complete path with wildcards

    def __getitem__(self, index):
        # 读取原始训练数据
        f = self.image_paths[index]
        pil_image = Image.open(f)
        pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.img_size)
        else:
            arr = center_crop_arr(pil_image, self.img_size)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        # 如果有目标文件夹路径，读取目标文件夹的前5张图像
        if self.target_folder and self.target_image_paths:
            target_image_path = self.target_image_paths[min(index, 4)]  # 选取前5张图像
            target_image = Image.open(target_image_path)
            target_image.load()
            target_image = target_image.convert("RGB")

            if self.random_crop:
                target_arr = random_crop_arr(target_image, self.img_size)
            else:
                target_arr = center_crop_arr(target_image, self.img_size)

            if self.random_flip and random.random() < 0.5:
                target_arr = target_arr[:, ::-1]

            target_arr = target_arr.astype(np.float32) / 127.5 - 1
        else:
            target_arr = None

        return np.transpose(arr, [2, 0, 1]), target_arr

    def __len__(self):
        return len(self.image_paths)


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

###################################################################


# class IMAGENET_dataset(Dataset):
#     def __init__(self, image_root, mode='val', class_num=None, img_size=512, random_crop=True, random_flip=False):
#         super().__init__()
#         if class_num is not None:
#             self.data_dir = os.path.join(image_root, mode, IMAGENET_DIC[str(class_num)][0], '*.JPEG')
#             self.image_paths = sorted(glob(self.data_dir))
#         else:
#             self.data_dir = os.path.join(image_root, mode, '*', '*.JPEG')
#             self.image_paths = sorted(glob(self.data_dir))
#         self.img_size = img_size
#         self.random_crop = random_crop
#         self.random_flip = random_flip
#         self.class_num = class_num

#     def __getitem__(self, index):
#         f = self.image_paths[index]
#         pil_image = Image.open(f)
#         pil_image.load()
#         pil_image = pil_image.convert("RGB")

#         if self.random_crop:
#             arr = random_crop_arr(pil_image, self.img_size)
#         else:
#             arr = center_crop_arr(pil_image, self.img_size)

#         if self.random_flip and random.random() < 0.5:
#             arr = arr[:, ::-1]

#         arr = arr.astype(np.float32) / 127.5 - 1

#         # y = [self.class_num, IMAGENET_DIC[str(self.class_num)][0], IMAGENET_DIC[str(self.class_num)][1]]
#         # y = self.class_num

#         return np.transpose(arr, [2, 0, 1])#, y

#     def __len__(self):
#         return len(self.image_paths)


# def center_crop_arr(pil_image, image_size):
#     # We are not on a new enough PIL to support the `reducing_gap`
#     # argument, which uses BOX downsampling at powers of two first.
#     # Thus, we do it by hand to improve downsample quality.
#     while min(*pil_image.size) >= 2 * image_size:
#         pil_image = pil_image.resize(
#             tuple(x // 2 for x in pil_image.size), resample=Image.BOX
#         )

#     scale = image_size / min(*pil_image.size)
#     pil_image = pil_image.resize(
#         tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
#     )

#     arr = np.array(pil_image)
#     crop_y = (arr.shape[0] - image_size) // 2
#     crop_x = (arr.shape[1] - image_size) // 2
#     return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


# def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
#     min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
#     max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
#     smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

#     # We are not on a new enough PIL to support the `reducing_gap`
#     # argument, which uses BOX downsampling at powers of two first.
#     # Thus, we do it by hand to improve downsample quality.
#     while min(*pil_image.size) >= 2 * smaller_dim_size:
#         pil_image = pil_image.resize(
#             tuple(x // 2 for x in pil_image.size), resample=Image.BOX
#         )

#     scale = smaller_dim_size / min(*pil_image.size)
#     pil_image = pil_image.resize(
#         tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
#     )

#     arr = np.array(pil_image)
#     crop_y = random.randrange(arr.shape[0] - image_size + 1)
#     crop_x = random.randrange(arr.shape[1] - image_size + 1)
#     return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
