from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
import torch
from torchvision import transforms

local_model_path = "models/sd-image-variations-diffusers"

device = "cuda:0"
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
  local_model_path,
  revision="v2.0",
  )
sd_pipe = sd_pipe.to(device)

im = Image.open("data/000.png")
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
    transforms.Normalize(
      [0.48145466, 0.4578275, 0.40821073],
      [0.26862954, 0.26130258, 0.27577711]),
])
inp = tform(im).to(device).unsqueeze(0)

out = sd_pipe(inp, guidance_scale=3)
out["images"][0].save("data/IVD_result2.jpg")
