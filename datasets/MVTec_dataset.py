from PIL import Image
from glob import glob
import os
from torch.utils.data import Dataset
import math
import numpy as np
import random

def get_all_good_data_paths(image_root, category_name=None):
    """获取所有 'good' 类别的图片路径"""
    if category_name is not None:
        data_dir = os.path.join(image_root, category_name, 'image', 'good', '*.png')
    else:
        data_dir = os.path.join(image_root, '*', 'image', 'good', '*.png')
    return sorted(glob(data_dir))


def get_all_mask_paths(image_root, category_name=None, mask_folder='mask'):
    """获取所有 mask 的路径"""
    if category_name is not None:
        data_dir = os.path.join(image_root, category_name, mask_folder, '*', '*.png')
    else:
        data_dir = os.path.join(image_root, '*', mask_folder, '*', '*.png')
    return sorted(glob(data_dir))

def get_target_mask_paths(image_root, category_name, target_folder):
    """获取目标文件夹对应的 mask 文件路径"""
    data_dir = os.path.join(image_root, category_name, 'mask', target_folder, '*.png')
    print(f"Searching for masks in: {data_dir}")
    return sorted(glob(data_dir))

def get_target_image_paths(image_root, category_name=None, target_folder=None, num_images=None):
    """
    根据 target_folder 获取目标文件夹中的随机 n 个图像路径
    """
    data_dir = os.path.join(image_root, category_name, 'image', target_folder, '*.png')
    print(f"Searching in: {data_dir}")
    
    image_paths = sorted(glob(data_dir))
    if num_images is None or num_images >= len(image_paths):
        return image_paths
    return random.sample(image_paths, num_images)


def split_data(data_list, split_ratio):
    """按比例划分数据"""
    split_index = int(len(data_list) * split_ratio)
    return data_list[:split_index], data_list[split_index:]


def get_mvtec_dataset(data_root, config, random_crop=False, random_flip=False, split_ratio=0.8):
    """
    获取训练、测试和（可选）目标数据集。

    参数:
        data_root (str): 数据集根目录。
        config: 配置对象，包含类别、图像尺寸、mask 和 target 参数等。
        random_crop (bool): 是否随机裁剪。
        random_flip (bool): 是否随机翻转。
        split_ratio (float): 训练集和测试集划分比例。

    返回:
        tuple: 包含训练集、测试集，以及（如果 config.data.use_target=True）目标数据集。
    """
    category_name = config.data.category
    target_folder = config.data.target_folder

    # 获取 "good" 图像路径
    good_image_paths = get_all_good_data_paths(data_root, category_name=category_name)
    train_image_paths = [good_image_paths[0]] if config.data.use_mask and good_image_paths else good_image_paths

    # 获取 target_folder 对应的 mask 路径并划分
    if config.data.use_mask:
        mask_paths = get_target_mask_paths(data_root, category_name, target_folder)
        train_mask_paths, test_mask_paths = split_data(mask_paths, split_ratio)
    else:
        train_mask_paths, test_mask_paths = None, None

    # 构建训练和测试数据集
    train_dataset = MVTec_dataset(
        train_image_paths, train_mask_paths,
        data_root, category_name, mode='train',
        img_size=config.data.image_size, random_crop=random_crop,
        random_flip=random_flip, use_mask=config.data.use_mask
    )
    test_dataset = MVTec_dataset(
        train_image_paths if config.data.use_mask else good_image_paths, test_mask_paths,
        data_root, category_name, mode='test',
        img_size=config.data.image_size, random_crop=random_crop,
        random_flip=random_flip, use_mask=config.data.use_mask
    )

    if config.data.use_target:
        target_image_paths = get_target_image_paths(data_root, category_name=category_name, target_folder=target_folder, num_images=5)
        target_dataset = MVTec_dataset(
            target_image_paths, None, data_root, category_name, target_folder=target_folder,
            mode='target', img_size=config.data.image_size, random_crop=random_crop,
            random_flip=random_flip, use_mask=False
        )
        return train_dataset, test_dataset, target_dataset

    return train_dataset, test_dataset


class MVTec_dataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_root, category_name, mode='train', img_size=512,
                 random_crop=True, random_flip=False, target_folder=None, use_mask=True):
        """
        初始化 MVTec 数据集。

        参数:
            image_paths (list): 图像路径列表。
            mask_paths (list): Mask 路径列表。
            image_root (str): 数据集根目录。
            category_name (str): 类别名称。
            mode (str): 数据集模式 ('train', 'test', 'target')。
            img_size (int): 输出图像尺寸。
            random_crop (bool): 是否随机裁剪。
            random_flip (bool): 是否随机翻转。
            target_folder (str): 目标文件夹名称。
            use_mask (bool): 是否加载和处理 mask。
        """
        super().__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_root = image_root
        self.category_name = category_name
        self.img_size = img_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.target_folder = target_folder
        self.use_mask = use_mask

    def __getitem__(self, index):
        # 加载图像
        image_path = self.image_paths[index % len(self.image_paths)]
        pil_image = Image.open(image_path).convert("RGB")

        # 加载 mask
        if self.use_mask and self.mask_paths:
            mask_path = self.mask_paths[index]
            pil_mask = Image.open(mask_path).convert("L")
        else:
            pil_mask = None

        # 随机裁剪或中心裁剪
        if self.random_crop:
            img_arr = random_crop_arr(pil_image, self.img_size)
            mask_arr = random_crop_arr(pil_mask, self.img_size) if pil_mask else None
        else:
            img_arr = center_crop_arr(pil_image, self.img_size)
            mask_arr = center_crop_arr(pil_mask, self.img_size) if pil_mask else None

        # 随机翻转
        if self.random_flip and random.random() < 0.5:
            img_arr = img_arr[:, ::-1]
            if mask_arr is not None:
                mask_arr = mask_arr[:, ::-1]

        # 归一化
        img_arr = img_arr.astype(np.float32) / 127.5 - 1
        if mask_arr is not None:
            mask_arr = mask_arr.astype(np.float32) / 255.0

        # 如果使用 mask，将 mask 拼接为额外通道
        if self.use_mask and mask_arr is not None:
            combined_arr = np.concatenate((np.transpose(img_arr, [2, 0, 1]), mask_arr[np.newaxis, ...]), axis=0)
        else:
            combined_arr = np.transpose(img_arr, [2, 0, 1])

        return combined_arr

    def __len__(self):
        return len(self.mask_paths) if self.use_mask else len(self.image_paths)


def split_data(data_list, split_ratio):
    """按比例划分数据"""
    split_index = int(len(data_list) * split_ratio)
    return data_list[:split_index], data_list[split_index:]


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
