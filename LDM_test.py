import sys
import torch
import numpy as np
from PIL import Image
from latent_diffusion.ldm.models.diffusion.ddim import DDIMSampler
import matplotlib.pyplot as plt
from torchvision import transforms
import yaml
from omegaconf import OmegaConf
from latent_diffusion.ldm.util import instantiate_from_config

# 配置模型路径和其他参数
model_path = "latent_diffusion/models/ldm/cin256-v2/model.ckpt"
config_path = "latent_diffusion/configs/latent-diffusion/cin256-v2.yaml"  # 配置文件路径
image_size = 256  # 图像尺寸
output_image_path = "generated_image.png"  # 保存生成图像的路径

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

# 创建模型（假设有一个工厂方法来处理模型创建）
def load_model(model_path, config):
    # 使用config来实例化模型
    model = load_model_from_config(config, model_path)  # 使用配置文件中的信息来构建模型
    
    # # 加载预训练模型
    # checkpoint = torch.load(model_path, map_location="cuda")  
    # state_dict = checkpoint["state_dict"]
    # model.load_state_dict(state_dict, strict=False)  # 加载权重
    # model = model.cuda()  # 将模型转移到GPU
    # model.eval()  # 切换到评估模式
    return model

# 加载和预处理输入图像
def load_and_preprocess_image(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0).cuda()  # 添加batch维度并移动到GPU
    return image

# 使用输入图像生成图像
def generate_image_from_input(model, input_image):
    # 将输入图像转为latent空间表示
    with torch.no_grad():
        latent_input = model.encode_first_stage(input_image)  # 假设有encode_first_stage方法
    
    # 使用DDIM采样器进行图像生成
    sampler = DDIMSampler(model)
    
    with torch.no_grad():
        # 用输入的latent图像生成新图像
        samples, _ = sampler.sample(S=50, conditioning=latent_input, batch_size=1, shape=[3, image_size, image_size],
                                    unconditional_guidance_scale=7.5, eta=0.0)
        
        # 解码生成的图像
        decoded_samples = model.decode_first_stage(samples)
        decoded_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)  # 归一化到[0, 1]
        return decoded_samples.squeeze(0).cpu().numpy()  # 移除batch维度并转为numpy数组

# 保存生成的图像
def save_image(image_array, save_path):
    image = np.transpose(image_array, (1, 2, 0))  # 将通道移到最后
    image = (image * 255).astype(np.uint8)  # 将像素值放大到[0, 255]范围
    pil_image = Image.fromarray(image)  # 转换为PIL图像
    pil_image.save(save_path)  # 保存图像

# 加载模型配置和实例化模型
def get_model():
    config = OmegaConf.load(config_path)  # 加载配置文件
    model = load_model(model_path, config)  # 根据配置加载模型
    return model

if __name__ == "__main__":
    # 加载模型
    model = get_model()
    
    # 加载并预处理输入图像
    input_image_path = "data/000.png"  # 输入图片的路径
    input_image = load_and_preprocess_image(input_image_path, image_size)
    output_image_path = "data/LDM_result.jpg"
    
    # 生成图像
    generated_image = generate_image_from_input(model, input_image)
    
    # 保存生成的图像
    save_image(generated_image, output_image_path)
    print(f"生成的图像已保存到 {output_image_path}")
