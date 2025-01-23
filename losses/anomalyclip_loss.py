import sys
import os
# Add the parent directory of 'losses' (or the directory containing 'Meta_Inspector') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from Meta_Inspector.AnomalyCLIP_lib import load, compute_similarity, get_similarity_map
from Meta_Inspector.utils import get_transform
from Meta_Inspector.prompt_ensemble import AnomalyCLIP_PromptLearner
from Meta_Inspector.AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
from PIL import Image
import torch.nn.functional as F
from torch import nn
import cv2

class AnomalyCLIPLoss(nn.Module):
    def __init__(self, model_path, features_list=[6, 12, 18, 24], 
                 image_size=518, n_ctx=12, t_n_ctx=4, depth=9, sigma=4, gt_mask_path=None):
        super(AnomalyCLIPLoss, self).__init__()

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device('cpu')
        self.features_list = features_list
        self.image_size = image_size
        self.sigma = sigma
        self.gt_mask_path = gt_mask_path

        # Load model
        self.AnomalyCLIP_parameters = {"Prompt_length": n_ctx, 
                                       "learnable_text_embedding_depth": depth, 
                                       "learnable_text_embedding_length": t_n_ctx}
        self.model, _ = load("ViT-L/14@336px", device=self.device, design_details=self.AnomalyCLIP_parameters)
        self.model.eval()

        # Load prompt learner
        self.prompt_learner = AnomalyCLIP_PromptLearner(self.model.to("cpu"), self.AnomalyCLIP_parameters)
        checkpoint = torch.load(model_path)
        self.prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        self.prompt_learner.to(self.device)
        self.model.to(self.device)
        self.model.visual.DAPM_replace(DPAM_layer=20)

        # Precompute prompts and text features
        prompts, tokenized_prompts, compound_prompts_text = self.prompt_learner(cls_id=None)
        self.text_features = self.model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        self.text_features = torch.stack(torch.chunk(self.text_features, dim=0, chunks=2), dim=1)
        self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        # Preprocessing transformation
        class Args:
            def __init__(self, image_size):
                self.image_size = image_size
        
        args = Args(image_size=image_size)
        self.preprocess, _ = get_transform(args)

        # Losses
        self.cross_entropy = nn.CrossEntropyLoss()

    def extract_feats(self, image):
        """Extract features from an image."""
        with torch.no_grad():
            image_features, patch_features = self.model.encode_image(image, self.features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features, patch_features

    def forward(self, image_tensor):
        """Compute both image-level and pixel-level losses."""
        label = torch.tensor([1])  # You can adjust the label depending on your task
        
        # Ensure the input is on the correct device
        image = image_tensor.to(self.device)
        
        # print(f"Loading mask from path: {self.gt_mask_path}")
        # gt_mask = cv2.imread(self.gt_mask_path, cv2.IMREAD_GRAYSCALE)  # 如果是灰度图
        # if gt_mask is None:
        #     print("Failed to load the mask image.")
        # else:
        #     print(f"Mask shape: {gt_mask.shape}")

        
        # Load ground truth mask (still read from file since it's separate from the image)
        gt_mask = cv2.imread(self.gt_mask_path, cv2.IMREAD_GRAYSCALE)  # Read the mask as grayscale
        gt_mask = cv2.resize(gt_mask, (self.image_size, self.image_size))  # Resize mask to match image size
        gt_mask = torch.tensor(gt_mask).unsqueeze(0).float().to(self.device)  # Add batch and channel dimensions
        
        # Extract features
        image_features, patch_features = self.extract_feats(image)
        
        # Image-level loss
        text_probs = image_features @ self.text_features.permute(0, 2, 1)
        text_probs = text_probs[:, 0, ...] / 0.07  # Shape: [batch_size, num_classes]
        image_loss = self.cross_entropy(text_probs, label.long().to(self.device))

        # Pixel-level similarity maps
        similarity_map_list = []
        for idx, patch_feature in enumerate(patch_features):
            if idx >= len(self.features_list):
                continue
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
            similarity, _ = compute_similarity(patch_feature, self.text_features[0])
            similarity_map = get_similarity_map(similarity[:, 1:, :], self.image_size).permute(0, 3, 1, 2)
            similarity_map_list.append(similarity_map)

        # Compute pixel-level loss
        pixel_loss = 0
        for similarity_map in similarity_map_list:
            pixel_loss += F.binary_cross_entropy_with_logits(similarity_map[:, 1, :, :], gt_mask)
            pixel_loss += F.binary_cross_entropy_with_logits(similarity_map[:, 0, :, :], 1 - gt_mask)

        # Aggregate similarity maps
        anomaly_map = torch.stack(similarity_map_list).sum(dim=0)
        anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=self.sigma)) 
                                for i in anomaly_map.detach().cpu()], dim=0)

        # Aggregate all losses into a single scalar
        total_loss = image_loss + pixel_loss

        return total_loss, anomaly_map.detach().cpu().numpy()



# Example usage
# image_path = "../Meta_Inspector/test_images/001.png"
# gt_mask_path = "../Meta_Inspector/gt_mask/000_mask.png"  # Path to ground truth mask
# model_path = "../Meta_Inspector/checkpoints/epoch_15.pth"
# anomaly_model = AnomalyCLIPLoss(model_path=model_path)

# # Simulated input
# label = torch.tensor([1])  # Image-level label (1 for anomaly)

# # Generate similarity map and compute losses
# losses, anomaly_map = anomaly_model(image_path, label, gt_mask_path)

# print(f"Losses: {losses}")

# # Save result
# import matplotlib.pyplot as plt
# visualization_map = anomaly_map[0, 1, :, :]  # Extract the anomaly probability channel
# plt.imshow(visualization_map, cmap='hot')
# plt.colorbar()
# output_path = "../Meta_Inspector/test_results/001.png"
# plt.savefig(output_path)
# plt.close()
# print(f"Similarity map saved to {output_path}")
