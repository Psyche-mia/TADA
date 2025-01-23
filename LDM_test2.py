import sys
# sys.path.append(".")
sys.path.append('/taming-transformers')
# from taming.models import vqgan
#@title loading utils
import torch
from omegaconf import OmegaConf

from latent_diffusion.ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("/root/TADA/latent_diffusion/configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "/root/TADA/latent_diffusion/models/ldm/cin256-v2/model.ckpt")
    return model

from latent_diffusion.ldm.models.diffusion.ddim import DDIMSampler

model = get_model()
sampler = DDIMSampler(model)

import torch
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

# Define classes and parameters
# classes = [25, 187, 448, 992]  # define classes to be sampled here
classes = [737]
n_samples_per_class = 6

ddim_steps = 20
ddim_eta = 0.0
scale = 3.0  # for unconditional guidance

all_samples = list()

# Ensure the model is loaded (replace with actual model loading code)
# model = <Your model loading code here>
# sampler = <Your sampler code here>

with torch.no_grad():
    with model.ema_scope():
        # Unconditional conditioning
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(n_samples_per_class * [1000]).to(model.device)}
        )
        
        # Generate samples for each class
        for class_label in classes:
            print(f"Rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
            xc = torch.tensor(n_samples_per_class * [class_label])
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

            # Sample using the sampler
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=n_samples_per_class,
                                             shape=[3, 64, 64],
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta)

            # Decode and clamp the samples
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            all_samples.append(x_samples_ddim)

# Combine all samples into a grid
grid = torch.stack(all_samples, 0)
grid = rearrange(grid, 'n b c h w -> (n b) c h w')
grid = make_grid(grid, nrow=n_samples_per_class)

# Convert grid to an image and save it
grid_image = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
grid_image_pil = Image.fromarray(grid_image.astype(np.uint8))

# Save the image to a file
grid_image_pil.save('data/generated_image_grid.png')

print("Image saved as 'generated_image_grid.png'")
