import time
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step, stable_diffusion_denoising_step
from losses import id_loss
from losses.anomalyclip_loss import AnomalyCLIPLoss
from losses.clip_loss import CLIPLoss, TargetLatentLoss
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS, HYBRID_MODEL_PATHS, HYBRID_CONFIG, ANOMALY_MODEL_PATHS
from datasets.imagenet_dic import IMAGENET_DIC
from utils.align_utils import run_alignment
from diffusers import StableDiffusionImageVariationPipeline
from torchvision import transforms
from torchvision.transforms import ToPILImage

class DiffusionCLIP(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        print(type(betas))  # 打印类型，确认是 numpy.ndarray
        print(betas.shape)  # 打印形状，确认其形状是否符合预期
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        if self.args.edit_attr is None:
            self.src_txts = self.args.src_txts
            self.trg_txts = self.args.trg_txts
        else:
            self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]
            self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]


    def clip_finetune_eff(self):
        # define tform
        tform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=False,
                ),
            transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]),])
        # print(f"Dataset: {self.config.data.dataset}")
        # print(f"Category: {self.config.data.category}")
        print(self.args.exp)
        print(f'   {self.src_txts}')
        print(f'-> {self.trg_txts}')

        # ----------- Model -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET", "MVTec"]:
            # print("model is ok")
            pass
        else:
            # print("error here 1")
            raise ValueError

        if self.config.model.type == "stable_diffusion":
            local_model_path = "pretrained/sd-image-variations-diffusers"
            model = StableDiffusionImageVariationPipeline.from_pretrained(local_model_path, revision="v2.0")
            print("Stable Diffusion Model loaded.")   
        elif self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET", "MVTec"]:
            model = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
            print("Improved diffusion Model loaded.")
        else:
            print('Not implemented dataset')
            raise ValueError
        # print(f"Load checkpoint from {url}")
        if self.config.model.type == "stable_diffusion":
            model.to(self.device)
            # Move scheduler tensors to the GPU
            for attr in dir(model.scheduler):
                tensor = getattr(model.scheduler, attr, None)
                if isinstance(tensor, torch.Tensor):
                    setattr(model.scheduler, attr, tensor.to(self.device))
            # model = torch.nn.DataParallel(model)
            
            # Check the total number of parameters in the model
            # total_params = sum(p.numel() for p in model.parameters())
            # print(f"Total parameters in the model: {total_params}")

            # # Check if any of the submodules have parameters
            # print("UNet parameters:", sum(p.numel() for p in model.module.unet.parameters()))
            # print("VAE parameters:", sum(p.numel() for p in model.module.vae.parameters()))
            # print("Text Encoder parameters:", sum(p.numel() for p in model.module.text_encoder.parameters()))
        else:
            model.load_state_dict(init_ckpt)
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        # ----------- Optimizer and Scheduler -----------#
        print(f"Setting optimizer with lr={self.args.lr_clip_finetune}")
        if self.config.model.type == "stable_diffusion":
            optim_ft = torch.optim.Adam(model.unet.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)
        else:
            optim_ft = torch.optim.Adam(model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)
        # optim_ft = torch.optim.SGD(model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)#, momentum=0.9)
        init_opt_ckpt = optim_ft.state_dict()
        scheduler_ft = torch.optim.lr_scheduler.StepLR(optim_ft, step_size=1, gamma=self.args.sch_gamma)
        init_sch_ckpt = scheduler_ft.state_dict()

        # ----------- Precompute Latents -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = self.args.bs_train
        img_lat_pairs_dic = {}

        for mode in ['train', 'test']:
            img_lat_pairs = []
            if self.config.model.type == "stable_diffusion":
                pairs_path = os.path.join('precomputed/',
                                          f'sd_{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
                print("pairs path for stable diffusion")   
            elif self.args.edit_attr in ['female', 'male']:
                self.config.data.dataset = 'GENDER'
                self.config.data.category = 'GENDER'
                if self.args.edit_attr == 'female':
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_male_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
                else:
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_female_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')

            elif self.config.data.dataset == "IMAGENET":
                if self.args.target_class_num is not None:
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_{IMAGENET_DIC[str(self.args.target_class_num)][1]}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
                else:
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')         
            else:
                pairs_path = os.path.join('precomputed/',
                                          f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path, map_location=self.device)
                for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                    tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                  f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                if self.args.edit_attr == 'female':
                    train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config,
                                                              gender='male')
                elif self.args.edit_attr == 'male':
                    train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config,
                                                              gender='female')
                else:
                    train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config,
                                                              target_class_num=self.args.target_class_num)
                    

                loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                                            num_workers=self.config.data.num_workers)
                loader = loader_dic[mode]

            # Unconditional text embeddings
            # uncond_embeddings = model.text_encoder(model.tokenizer([""], return_tensors="pt").input_ids.to(self.device))[0]
            for step, img in enumerate(loader):
                if self.config.model.type == "stable_diffusion":
                    to_pil = ToPILImage()
                    print("image shape: ", img.shape)
                    # convert rgba to rgb and convert tensor to PIL then process the transform and convert back to tensor
                    x0 = tform(to_pil(img.squeeze(0)[:3, :, :])).to(self.config.device).unsqueeze(0)
                    # Save the image in 'result/' directory
                    # pil_img = to_pil(img.squeeze(0)[:3, :, :])
                    # save_path = "data/converted_image.png"
                    # pil_img.save(save_path)
                else:
                    x0 = img.to(self.config.device)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                # Stable diffusion model to evaluation mode
                model.unet.eval()
                model.vae.eval()
                model.scheduler.set_timesteps(num_inference_steps=self.args.n_inv_step)
                time_s = time.time()
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            # print(type(t))
                            t_prev = (torch.ones(n) * j).to(self.device)
                            
                            # add stabel diffusion method
                            if self.config.model.type == "stable_diffusion":
                                # Check if the current input is an image or latent
                                x = stable_diffusion_denoising_step(model, x, t.long(), device=self.device, is_latent=(it > 0))
                                # Encode the image into latent space using the VAE
                                # x = model.vae.encode(x).latent_dist.sample()
                                # print("x shape before scaling: ", x.shape)
                                # # print(type(t))
                                # x = x * model.vae.config.scaling_factor  # Scale latent
                                # print("x shape after scaling: ", x.shape)
                                # # Denoising step
                                # # Example latent tensor x (e.g., after passing through the VAE)
                                # batch_size, _, height, width = x.shape
                                # seq_length = height * width  # Latent tokens, e.g., 784

                                # # Create dummy embeddings with the correct shape
                                # dummy_embeddings = torch.zeros((batch_size, seq_length, 768), device=x.device)  # Use 768, not 320

                                # # Print the shape to verify
                                # print(f"Dummy embeddings shape: {dummy_embeddings.shape}")
                                # # Pass to UNet
                                # noise_pred = model.unet(x, t, encoder_hidden_states=dummy_embeddings)["sample"]
                                # # t = t.long()  # Cast to long
                                # print(type(t))
                                # # print("noise prediction", noise_pred)
                                # x = model.scheduler.step(noise_pred, t, x)["prev_sample"]
                            else:
                                x = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma)

                            progress_bar.update(1)
                    time_e = time.time()
                    print(f'{time_e - time_s} seconds')
                    print("latent x:", x)
                    x_lat = x.clone()
                    tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        time_s = time.time()
                        # seq_inv and seq_inv_next are reversed
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)
                            
                            # add stabel diffusion method
                            if self.config.model.type == "stable_diffusion":
                                # Denoising step
                                # If it's the last step, mark as final step
                                is_final_step = (it == len(seq_inv) - 1)
                                
                                x = stable_diffusion_denoising_step(model, x, t.long(), self.device, is_latent=True, is_final_step=is_final_step)
                            else:
                                x = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma)
                            progress_bar.update(1)
                        time_e = time.time()
                        print(f'{time_e - time_s} seconds')

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone()])
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            torch.save(img_lat_pairs, pairs_path)
            
        # ----------- Loss -----------#
        print("Loading losses")
        clip_loss_func = CLIPLoss(
            self.device,
            lambda_direction=1,
            lambda_patch=0,
            lambda_global=0,
            lambda_manifold=0,
            lambda_texture=0,
            clip_model=self.args.clip_model_name)
        # target_latent_loss_func = TargetLatentLoss(
        #     device=self.device,
        #     target_latents=img_lat_pairs_dic['target'],  # 传入 target latent 的字典
        #     latent_loss_type=self.args.latent_loss_type)  # 可选 'cosine' 或 'l1'
        anomaly_loss_func = AnomalyCLIPLoss(model_path=ANOMALY_MODEL_PATHS[0], gt_mask_path=self.config.path.gt_mask_path)

        if self.config.data.dataset == "CelebA_HQ":
            id_loss_func = id_loss.IDLoss().to(self.device).eval()
        else:
            id_loss_func = None

        # ----------- Finetune Diffusion Models -----------#
        print("Start finetuning")
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print('Uniform skip type')
        else:
            seq_train = list(range(self.args.t_0))
            print('No skip')
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])

        for src_txt, trg_txt in zip(self.src_txts, self.trg_txts):
            print(f"CHANGE {src_txt} TO {trg_txt}")
            if self.config.model.type == "stable_diffusion":
                pass
            else:
                model.module.load_state_dict(init_ckpt)
            optim_ft.load_state_dict(init_opt_ckpt)
            scheduler_ft.load_state_dict(init_sch_ckpt)
            clip_loss_func.target_direction = None

            # ----------- Train -----------#
            for it_out in range(self.args.n_iter):
                exp_id = os.path.split(self.args.exp)[-1]
                save_name = f'checkpoint/{exp_id}-{it_out}.pth'
                if self.args.do_train:
                    if os.path.exists(save_name):
                        print(f'{save_name} already exists.')
                        model.module.load_state_dict(torch.load(save_name))
                        continue
                    else:
                        for step, (x0, _, x_lat) in enumerate(img_lat_pairs_dic['train']):
                            if self.config.model.type == "stable_diffusion":
                                model.unet.train()
                            else:
                                model.train()
                            time_in_start = time.time()

                            optim_ft.zero_grad()
                            x = x_lat.clone().to(self.device)
                            x0 = x0.to(self.device)
                            with tqdm(total=len(seq_train), desc=f"CLIP iteration") as progress_bar:
                                for t_it, (i, j) in enumerate(zip(reversed(seq_train), reversed(seq_train_next))):
                                    t = (torch.ones(n) * i).to(self.device)
                                    t_next = (torch.ones(n) * j).to(self.device)
                                    if self.config.model.type == "stable_diffusion":
                                        # Denoising step
                                        # If it's the last step, mark as final step
                                        is_final_step = (t_it == len(seq_inv) - 1)
                                        print("training latent", x)
                                        
                                        x = stable_diffusion_denoising_step(model, x, t.long(), self.device, is_latent=True, is_final_step=is_final_step)
                                    else:
                                        x, x0_t = denoising_step(x, t=t, t_next=t_next, models=model,
                                                                logvars=self.logvar,
                                                                sampling_type=self.args.sample_type,
                                                                b=self.betas,
                                                                eta=self.args.eta,
                                                                learn_sigma=learn_sigma,
                                                                out_x0_t=True)
                                    progress_bar.update(1)
                                    x = x.detach().clone()

                                    loss_clip = -torch.log((2 - clip_loss_func(x0, src_txt, x0_t, trg_txt)) / 2)
                                    loss_l1 = nn.L1Loss()(x0, x0_t)
                                    loss = self.args.clip_loss_w * loss_clip + self.args.l1_loss_w * loss_l1
                                    
                                    # anomaly loss
                                    anomaly_loss, _ = anomaly_loss_func(x0_t)
                                    loss += self.args.anomalyclip_loss_w * anomaly_loss
                                    
                                    # Add Target Latent Loss
                                    # target_latent_loss = target_latent_loss_func(x)  # Calculate target latent loss
                                    # loss += self.args.target_latent_loss_w * target_latent_loss  # Add target latent loss to the overall loss
                                    if self.config.data.dataset == "CelebA_HQ":
                                        loss_id = torch.mean(id_loss_func(x0, x))
                                        loss += self.args.id_loss_w * loss_id
                                    
                                    # Check GPU memory before the backward pass
                                    # print("Before backward pass:")
                                    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
                                    loss.backward(retain_graph=True)
                                    # Check GPU memory before the optimizer step
                                    # print("Before optimizer step:")
                                    # print(torch.cuda.memory_summary(device=None, abbreviated=False))


                                    optim_ft.step()
                                    if self.config.model.type == "stable_diffusion":
                                        for p in model.parameters():
                                            p.grad = None
                                    else:
                                        for p in model.module.parameters():
                                            p.grad = None
                                    print(f"CLIP {step}-{it_out}: loss_clip: {loss_clip:.3f}")
                                    # break

                            if self.args.save_train_image:
                                tvu.save_image((x0_t + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                              f'train_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_train_step}.png'))
                            time_in_end = time.time()
                            print(f"Training for 1 image takes {time_in_end - time_in_start:.4f}s")
                            if step == self.args.n_train_img - 1:
                                break

                        if isinstance(model, nn.DataParallel):
                            torch.save(model.module.state_dict(), save_name)
                        else:
                            torch.save(model.state_dict(), save_name)
                        print(f'Model {save_name} is saved.')
                        scheduler_ft.step()

                # ----------- Eval -----------#
                if self.args.do_test:
                    if not self.args.do_train:
                        print(save_name)
                        model.module.load_state_dict(torch.load(save_name))

                    model.eval()
                    img_lat_pairs = img_lat_pairs_dic[mode]
                    for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs):
                        with torch.no_grad():
                            x = x_lat.clone().to(self.device)
                            x0 = x0.to(self.device)
                            with tqdm(total=len(seq_test), desc=f"Eval iteration") as progress_bar:
                                for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                                    t = (torch.ones(n) * i).to(self.device)
                                    t_next = (torch.ones(n) * j).to(self.device)

                                    x = denoising_step(x, t=t, t_next=t_next, models=model,
                                                       logvars=self.logvar,
                                                       sampling_type=self.args.sample_type,
                                                       b=self.betas,
                                                       eta=self.args.eta,
                                                       learn_sigma=learn_sigma)

                                    progress_bar.update(1)

                            print(f"Eval {step}-{it_out}")
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'{mode}_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_test_step}.png'))
                            if step == self.args.n_test_img - 1:
                                break

    def clip_latent_optim(self):
        # ----------- Data -----------#
        n = 1

        if self.args.align_face and self.config.data.dataset in ["FFHQ", "CelebA_HQ"]:
            try:
                img = run_alignment(self.args.img_path, output_size=self.config.data.image_size)
            except:
                img = Image.open(self.args.img_path).convert("RGB")
        else:
            img = Image.open(self.args.img_path).convert("RGB")
        img = img.resize((self.config.data.image_size, self.config.data.image_size), Image.ANTIALIAS)
        img = np.array(img) / 255
        img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(dim=0)
        img = img.to(self.config.device)
        tvu.save_image(img, os.path.join(self.args.image_folder, f'0_orig.png'))
        x0 = (img - 0.5) * 2.

        # ----------- Model -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            pass
        else:
            raise ValueError

        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                ckpt = torch.load(self.args.model_path)
            else:
                ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET", "MVTec"]:
            model = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                ckpt = torch.load(self.args.model_path)
            else:
                ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
            print("Improved diffusion Model loaded.")
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.eval()

        # ----------- Loss -----------#
        print("Loading losses")
        id_loss_func = id_loss.IDLoss().to(self.device).eval()
        clip_loss_func = CLIPLoss(
            self.device,
            lambda_direction=0,
            lambda_patch=0,
            lambda_global=1,
            lambda_manifold=0,
            lambda_texture=0,
            clip_model=self.args.clip_model_name)

        # ----------- Invert Image to Latent -----------#
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])
        print(f"Finding latent")
        with torch.no_grad():
            x = x0.clone()

            with tqdm(total=len(seq_inv), desc=f"Inversion process") as progress_bar:
                for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                    t = (torch.ones(n) * i).to(self.device)
                    t_prev = (torch.ones(n) * j).to(self.device)

                    x = denoising_step(x, t=t, t_next=t_prev, models=model,
                                       logvars=self.logvar,
                                       sampling_type='ddim',
                                       b=self.betas,
                                       eta=0,
                                       learn_sigma=learn_sigma)

                    progress_bar.update(1)

            x_lat0 = x.clone()
            tvu.save_image((x_lat0 + 1) * 0.5, os.path.join(self.args.image_folder,
                                                            f'1_lat_ninv{self.args.n_inv_step}.png'))

        # ----------- Latent Optimization -----------#
        print(f"CLIP loss latent optimization")
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print('Uniform skip type')
        else:
            seq_train = list(range(self.args.t_0))
            print('No skip')
        seq_train_next = [-1] + list(seq_train[:-1])

        for txt_idx, (src_txt, trg_txt) in enumerate(zip(self.src_txts, self.trg_txts)):
            x_lat = nn.Parameter(x_lat0.clone())
            optimizer = torch.optim.Adam([x_lat], weight_decay=0, lr=self.args.lr_clip_lat_opt)
            clip_loss_func.target_direction = None

            with torch.set_grad_enabled(True):
                for it in range(self.args.n_iter):
                    x = x_lat
                    optimizer.zero_grad()
                    with tqdm(total=len(seq_train), desc=f"Generative process {trg_txt}-{it}") as progress_bar:
                        for i, j in zip(reversed(seq_train), reversed(seq_train_next)):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               eta=self.args.eta,
                                               learn_sigma=learn_sigma)
                            progress_bar.update(1)

                    loss_clip = (2 - clip_loss_func(x0, src_txt, x, trg_txt)) / 2
                    loss_clip = -torch.log(loss_clip)
                    loss_id = torch.mean(id_loss_func(x0, x))
                    loss_l1 = nn.L1Loss()(x0, x)

                    loss = self.args.clip_loss_w * loss_clip + self.args.id_loss_w * loss_id + self.args.l1_loss_w * loss_l1
                    loss.backward()
                    print(f"CLIP opt: loss_clip: {loss_clip:.3f}, loss_id: {loss_id:.3f}, loss_l1: {loss_l1:.3f}")
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                               f'2_clip_{trg_txt.replace(" ", "_")}_t{self.args.t_0}_{it}_ngen{self.args.n_train_step}.png'))

                    optimizer.step()

    def edit_images_from_dataset(self):
        # ----------- Models -----------#
        print(self.args.exp)
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            pass
        else:
            raise ValueError

        models = []
        model_paths = [None, self.args.model_path]
        for model_path in model_paths:
            if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
                model_i = DDPM(self.config)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
                learn_sigma = False
            elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET", "MVTec"]:
                model_i = i_DDPM(self.config.data.dataset)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
                learn_sigma = True
            else:
                print('Not implemented dataset')
                raise ValueError
            model_i.load_state_dict(ckpt)
            model_i.to(self.device)
            model_i = torch.nn.DataParallel(model_i)
            model_i.eval()
            print(f"{model_path} is loaded.")
            models.append(model_i)

        # ----------- Precompute Latents thorugh Inversion Process -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])
        n = 1
        img_lat_pairs_dic = {}
        for mode in ['test']:
            img_lat_pairs = []
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')

            if os.path.exists(pairs_path):
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path)
                for step, (x0, x_id, e_id) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                    tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_1_rec.png'))
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config)
                loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                                            num_workers=self.config.data.num_workers)
                loader = loader_dic[mode]

            for step, img in enumerate(loader):
                x0 = img.to(self.config.device)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=models,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma,
                                               ratio=0)

                            progress_bar.update(1)

                    x_lat = x.clone()
                    tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_next, models=models,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               eta=self.args.eta,
                                               learn_sigma=learn_sigma,
                                               ratio=0)

                            progress_bar.update(1)

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone()])

                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_1_rec.png'))
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)


        # ----------- Generative Process -----------#
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        if self.args.n_test_step != 0:
            seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
            seq_test = [int(s) for s in list(seq_test)]
            print('Uniform skip type')
        else:
            seq_test = list(range(self.args.t_0))
            print('No skip')
        seq_test_next = [-1] + list(seq_test[:-1])
        print("Start evaluation")
        eval_modes = ['test']
        for mode in eval_modes:

            img_lat_pairs = img_lat_pairs_dic[mode]
            for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs):

                with torch.no_grad():
                    x = x_lat
                    with tqdm(total=len(seq_test), desc=f"Eval iteration") as progress_bar:
                        for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_next, models=models,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               eta=self.args.eta,
                                               learn_sigma=learn_sigma,
                                               ratio=self.args.model_ratio,
                                               hybrid=self.args.hybrid_noise,
                                               hybrid_config=HYBRID_CONFIG)

                            progress_bar.update(1)

                    print(f"Eval {step}")
                    tvu.save_image((x + 1) * 0.5,
                                   os.path.join(self.args.image_folder,
                                                f'{mode}_{step}_2_clip_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}.png'))


    def edit_one_image(self):
        # ----------- Data -----------#
        n = self.args.bs_test

        if self.args.align_face and self.config.data.dataset in ["FFHQ", "CelebA_HQ"]:
            try:
                img = run_alignment(self.args.img_path, output_size=self.config.data.image_size)
            except:
                img = Image.open(self.args.img_path).convert("RGB")
        else:
            img = Image.open(self.args.img_path).convert("RGB")
        img = img.resize((self.config.data.image_size, self.config.data.image_size), Image.LANCZOS)
        img = np.array(img)/255
        img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(dim=0).repeat(n, 1, 1, 1)
        img = img.to(self.config.device)
        tvu.save_image(img, os.path.join(self.args.image_folder, f'0_orig.png'))
        x0 = (img - 0.5) * 2.

        # ----------- Models -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            pass
        else:
            raise ValueError

        models = []

        if self.args.hybrid_noise:
            model_paths = [None] + HYBRID_MODEL_PATHS
        else:
            model_paths = [None, self.args.model_path]

        for model_path in model_paths:
            if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
                model_i = DDPM(self.config)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
                learn_sigma = False
            elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET", "MVTec"]:
                model_i = i_DDPM(self.config.data.dataset)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
                learn_sigma = True
            else:
                print('Not implemented dataset')
                raise ValueError
            model_i.load_state_dict(ckpt)
            model_i.to(self.device)
            model_i = torch.nn.DataParallel(model_i)
            model_i.eval()
            print(f"{model_path} is loaded.")
            models.append(model_i)

        with torch.no_grad():
            #---------------- Invert Image to Latent in case of Deterministic Inversion process -------------------#
            if self.args.deterministic_inv:
                x_lat_path = os.path.join(self.args.image_folder, f'x_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}.pth')
                if not os.path.exists(x_lat_path):
                    seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
                    seq_inv = [int(s) for s in list(seq_inv)]
                    seq_inv_next = [-1] + list(seq_inv[:-1])

                    x = x0.clone()
                    with tqdm(total=len(seq_inv), desc=f"Inversion process ") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=models,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma,
                                               ratio=0,
                                               )

                            progress_bar.update(1)
                        x_lat = x.clone()
                        torch.save(x_lat, x_lat_path)
                else:
                    print('Latent exists.')
                    x_lat = torch.load(x_lat_path)


            # ----------- Generative Process -----------#
            print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, "
                  f" Steps: {self.args.n_test_step}/{self.args.t_0}")
            if self.args.n_test_step != 0:
                seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
                seq_test = [int(s) for s in list(seq_test)]
                print('Uniform skip type')
            else:
                seq_test = list(range(self.args.t_0))
                print('No skip')
            seq_test_next = [-1] + list(seq_test[:-1])

            for it in range(self.args.n_iter):
                if self.args.deterministic_inv:
                    x = x_lat.clone()
                else:
                    e = torch.randn_like(x0)
                    a = (1 - self.betas).cumprod(dim=0)
                    x = x0 * a[self.args.t_0 - 1].sqrt() + e * (1.0 - a[self.args.t_0 - 1]).sqrt()
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'1_lat_ninv{self.args.n_inv_step}.png'))

                with tqdm(total=len(seq_test), desc="Generative process {}".format(it)) as progress_bar:
                    for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)

                        x = denoising_step(x, t=t, t_next=t_next, models=models,
                                           logvars=self.logvar,
                                           sampling_type=self.args.sample_type,
                                           b=self.betas,
                                           eta=self.args.eta,
                                           learn_sigma=learn_sigma,
                                           ratio=self.args.model_ratio,
                                           hybrid=self.args.hybrid_noise,
                                           hybrid_config=HYBRID_CONFIG)

                        # added intermediate step vis
                        if (i - 99) % 100 == 0:
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'2_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_{i}_it{it}.png'))
                        progress_bar.update(1)

                x0 = x.clone()
                if self.args.model_path:
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                               f"3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}_{self.args.model_path.split('/')[-1].replace('.pth','')}.png"))
                else:
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}.png'))

    def unseen2unseen(self):
        # ----------- Data -----------#
        n = self.args.bs_test
        # if self.args.align_face and self.config.data.dataset in ["FFHQ", "CelebA_HQ"]:
        #     try:
        #         img = run_alignment(self.args.img_path, output_size=self.config.data.image_size)
        #     except:
        #         img = Image.open(self.args.img_path).convert("RGB")
        # else:
        img = Image.open(self.args.img_path).convert("RGB")
        img = img.resize((self.config.data.image_size, self.config.data.image_size), Image.ANTIALIAS)
        img = np.array(img) / 255
        img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(dim=0)
        img = img.to(self.config.device)
        tvu.save_image(img, os.path.join(self.args.image_folder, f'0_orig.png'))
        img = img.repeat(n, 1, 1, 1)
        x0 = (img - 0.5) * 2.

        # ----------- Models -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset == "AFHQ":
            pass
        else:
            raise ValueError

        models = []
        model_paths = [None, self.args.model_path]
        for model_path in model_paths:
            if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
                model_i = DDPM(self.config)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
                learn_sigma = False
            elif self.config.data.dataset in ["FFHQ", "AFHQ"]:
                model_i = i_DDPM(self.config.data.dataset)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
                learn_sigma = True
            else:
                print('Not implemented dataset')
                raise ValueError
            model_i.load_state_dict(ckpt)
            model_i.to(self.device)
            model_i = torch.nn.DataParallel(model_i)
            model_i.eval()
            print(f"{model_path} is loaded.")
            models.append(model_i)

        with torch.no_grad():
            print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
            print('Uniform skip type')
            for it in range(self.args.n_iter):
                seq_test = np.linspace(0, 1, 6) * self.args.t_0
                seq_test = [int(s) for s in list(seq_test)]
                seq_test_next = [-1] + list(seq_test[:-1])

                # ----------- Stochastic Foward Process -----------#
                e = torch.randn_like(x0)
                a = (1 - self.betas).cumprod(dim=0)
                x = x0 * a[self.args.t_0 - 1].sqrt() + e * (1.0 - a[self.args.t_0 - 1]).sqrt()
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder, f'1_lat.png'))

                # ----------- Generative Process -----------#
                with tqdm(total=len(seq_test), desc="Generative process {}".format(it)) as progress_bar:
                    for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)

                        x = denoising_step(x, t=t, t_next=t_next, models=models,
                                           logvars=self.logvar,
                                           sampling_type=self.args.sample_type,
                                           b=self.betas,
                                           eta=0,
                                           learn_sigma=learn_sigma,
                                           ratio=0)

                        progress_bar.update(1)

                x0 = x.clone()

                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'2_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}.png'))

            # ---------------- Invert Image to Latent through Detriministic Process -------------------#
            seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
            seq_inv = [int(s) for s in list(seq_inv)]
            seq_inv_next = [-1] + list(seq_inv[:-1])

            x = x0.clone()
            with tqdm(total=len(seq_inv), desc=f"Inversion process ") as progress_bar:
                for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                    t = (torch.ones(n) * i).to(self.device)
                    t_prev = (torch.ones(n) * j).to(self.device)

                    x = denoising_step(x, t=t, t_next=t_prev, models=models,
                                       logvars=self.logvar,
                                       sampling_type='ddim',
                                       b=self.betas,
                                       eta=0,
                                       learn_sigma=learn_sigma,
                                       ratio=0)

                    progress_bar.update(1)
                x_lat = x.clone()
                tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                               f'3_lat_ninv{self.args.n_inv_step}.png'))

            #----------- Generative Process with Finetuned Model -----------#
            print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, "
                  f" Steps: {self.args.n_test_step}/{self.args.t_0}")
            if self.args.n_test_step != 0:
                seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
                seq_test = [int(s) for s in list(seq_test)]
                print('Uniform skip type')
            else:
                seq_test = list(range(self.args.t_0))
                print('No skip')
            seq_test_next = [-1] + list(seq_test[:-1])

            x = x_lat.clone()

            with tqdm(total=len(seq_test), desc="Generative process {}".format(it)) as progress_bar:
                for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                    t = (torch.ones(n) * i).to(self.device)
                    t_next = (torch.ones(n) * j).to(self.device)

                    x = denoising_step(x, t=t, t_next=t_next, models=models,
                                       logvars=self.logvar,
                                       sampling_type=self.args.sample_type,
                                       b=self.betas,
                                       eta=self.args.eta,
                                       learn_sigma=learn_sigma,
                                       ratio=self.args.model_ratio)

                    progress_bar.update(1)

            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                       f'4_gen_t{self.args.t_0}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}.png'))