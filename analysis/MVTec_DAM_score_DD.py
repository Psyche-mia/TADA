import clip
import torch
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
text_type = 'defect description'

# 加载CLIP模型和预处理工具
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 定义图片文件夹路径和 captions.xlsx 文件路径
base_folder = "/mnt/IAD_datasets/Defect_Spectrum/DS-MVTec/bottle/image/"
captions_path = "/mnt/IAD_datasets/Defect_Spectrum/DS-MVTec/captions.xlsx"

# 加载 captions.xlsx
captions_df = pd.read_excel(captions_path)

# 提取 path 和 defect_description
captions_df['file_name'] = captions_df['Path'].apply(lambda x: os.path.basename(x))
image_to_caption = dict(zip(captions_df['file_name'], captions_df[text_type]))

# 初始化存储图片和标签的列表
image_features_list = []
text_features_list = []
image_labels = []
image_names = []

# 处理每个文件夹中的图片
for folder in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder)
    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, img_name)
            # if img_name not in image_to_caption:
            #     continue  # 跳过没有 defect description 的图片
            caption = image_to_caption[img_name] if img_name in image_to_caption and pd.notna(image_to_caption[img_name]) else "This is a good image."
            caption = image_to_caption[img_name]
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_feature = model.encode_image(image).cpu().numpy()
                image_features_list.append(image_feature)
            image_names.append((folder, img_name))
            image_labels.append(caption)  # 保存 defect description

# 将图片和文本的特征向量转为矩阵
image_features = torch.tensor(image_features_list).squeeze(1)  # (num_images, feature_dim)

# 处理 defect descriptions
for caption in image_labels:
    text_token = clip.tokenize([caption]).to(device)
    with torch.no_grad():
        text_feature = model.encode_text(text_token).cpu().numpy()
        text_features_list.append(text_feature)

# 将文本特征转为矩阵
text_features = torch.tensor(text_features_list).squeeze(1)  # (num_images, feature_dim)

# 对特征向量进行归一化
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# 计算图片与所有文本的余弦相似度
similarity_matrix = cosine_similarity(image_features, text_features)

# 初始化判别能力的分数列表
log_discrimination_scores = []

# 计算每张图片的判别能力分数 (引入对数函数)
for i, (folder, img_name) in enumerate(image_names):
    s_correct = similarity_matrix[i][i]  # 图片与其对应 defect description 的相似度
    s_incorrect = [similarity_matrix[i][j] for j in range(len(image_names)) if j != i]  # 与其他描述的相似度
    s_incorrect_mean = np.mean(s_incorrect)  # 非所属 defect description 的均值
    score = s_correct - s_incorrect_mean  # 判别能力分数
    # log_score = np.log1p(score)  # 使用 log(1 + x) 调整分数
    log_score = np.exp(s_correct - s_incorrect_mean) - 1
    log_discrimination_scores.append((folder, log_score))

# 按文件夹 (folder) 组织 DAM 分数
folder_dam_scores = {}
for folder, log_score in log_discrimination_scores:
    if folder not in folder_dam_scores:
        folder_dam_scores[folder] = []
    folder_dam_scores[folder].append(log_score)

# 计算每个文件夹的平均 DAM
folder_mean_dam = {folder: np.mean(scores) for folder, scores in folder_dam_scores.items() if scores}

# 绘制平均 DAM 柱状图
folders = list(folder_mean_dam.keys())
mean_dams = list(folder_mean_dam.values())

plt.figure(figsize=(10, 6))
plt.bar(folders, mean_dams, color='skyblue', edgecolor='black')
plt.title("Average Log-DAM by Folder", fontsize=14)
plt.xlabel("Folder (Label)", fontsize=12)
plt.ylabel("Average Log-DAM", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 保存图像

plt.savefig(f"average_log_dam_by_folder_{text_type}.png")
plt.show()
