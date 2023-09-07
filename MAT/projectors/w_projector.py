# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""
from utils.models_utils import toogle_grad, load_old_G
import copy
import torchvision.transforms as T
import wandb
import numpy as np
import torch
from loss import IDLoss
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import PIL.Image
from configs import global_config, hyperparameters
from utils import log_utils
import dnnlib
from PIL import Image
mpath = './segmentation'
#mask_list = sorted(glob.glob(mpath + '/*.png') + glob.glob(mpath + '/*.jpg') + glob.glob(mpath + '/*.JPG'))
mask = cv2.imread('./segmentation/3.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 
#print("mask shape",  mask.shape)
mask = cv2.resize(mask, dsize=(512,512))

mask = torch.from_numpy(mask).float().to(global_config.device).unsqueeze(0).view(1, 1, 512, 512)
original_G = load_old_G()
#print("mask shape", mask.shape)
mask = (mask * -1) + 1
#print(mask)
def project(
        G,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=False,
        device: torch.device,
        use_wandb=False,
        initial_w=None,
        image_log_step=global_config.image_rec_result_log_snapshot,
        w_name: str
):
    #print("projecting image to get w")
    #print("ttttttttt", target)
    #print(G.img_channels, G.img_resolution, G.img_resolution)
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_avg_tensor = torch.from_numpy(w_avg).to(global_config.device)
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    #targ = (target / 127.5) - 1
    targ = target / 255
    #print("################" ,targ)
    max_to_norm = torch.max(target)
    targ2 = (target / (max_to_norm / 2)) - 1
    if target_images.shape[2] > 256:

        

        target_images = F.interpolate(target_images, size=(256, 256), mode='area') / 255
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device,
                         requires_grad=True)  # pylint: disable=not-callable
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999),
                                 lr=hyperparameters.first_inv_lr)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True


    ls = IDLoss()
    for step in tqdm(range(num_steps)):

        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1]) 
        #mask = torch.zeros(targ.shape[0], 1, 512, 512).to(device) 
        #print(mask)
        #print("what should mask be in w_projector", targ)
        synth_images = G.synthesis(targ , mask, ws, noise_mode='const')
        #print(ws)
        s_image = ((synth_images.permute(0, 2, 3, 1) * 255).round()).clamp(0, 255).to(torch.uint8)
        if step % 5 == 0:
            PIL.Image.fromarray(s_image[0].cpu().numpy(), 'RGB').save('./see/proj' + str(step) + '.png')
        #print("targggggggggggggggggggggggggggggggggggg shape", targ.shape)
        transform = T.ToPILImage()
        tgimg = transform(targ)
        tarp = ((targ * 255).round()).clamp(0, 255).to(torch.uint8)
        tgimg.save('./see/targ' + str(step) + '.png')
        

        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()
        fnet_synthetic = ls.extract_feats((synth_images + 1) / 2)
        fnet_target = ls.extract_feats(targ.unsqueeze(0).to(device).to(torch.float32))
        #print(target_images)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        #print("minimizing this dist w proj", dist)
        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight
        #loss = (1 - (cos(fnet_synthetic, fnet_target)))  +  reg_loss * 1e4

        if step % image_log_step == 0:
            with torch.no_grad():
                if use_wandb:
                    global_config.training_step += 1
                    wandb.log({f'first projection _{w_name}': loss.detach().cpu()}, step=global_config.training_step)
                    log_utils.log_image_from_w(w_opt.repeat([1, G.mapping.num_ws, 1]), G, w_name)

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    del G
    #print(w_opt)
    return w_opt.repeat([1, 12, 1])
