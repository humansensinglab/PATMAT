# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
import cv2
from criteria import l2_loss
import pyspng
import imageio
import glob
import os
import lpips
from projectors import w_projector
from utils.models_utils import toogle_grad, load_old_G, load_old_D
from tqdm import tqdm
import re
import random
from loss import IDLoss
from networks.basic_module import FullyConnectedLayer, Conv2dLayer, MappingNet
from typing import List, Optional
from torch.nn import functional as F
import click
from numpy.linalg import norm
from numpy import dot
import clip
import dnnlib
import math
import lpips
import pickle
import numpy as np
from skimage.transform import resize, rescale
from PIL import Image, ImageDraw, ImageOps
import PIL.Image
import torchvision
import torch
import torch.nn.functional as F
from torch import optim
import legacy
#from datasets.mask_generator_512 import RandomMask
from networks.mat import Generator

device = torch.device('cuda')


def calc_inversion(G, image, image_name):
        id_image = torch.squeeze((image + 1) / 2) * 255
        w = w_projector.project(G, id_image, device=device, w_avg_samples=600,
                                    num_steps=200, w_name=image_name,
                                    use_wandb=False)

        return w


def num_range(s :str) -> List[int]:

    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dpath', help='the path of the input image', required=True)
@click.option('--refpath', help='the path of the ask')
@click.option('--mpath', help='the path of the ask')
@click.option('--resolution', type=int, help='resolution of input image', default=512, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='random', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')




def generate_images(
    ctx: click.Context,
    network_pkl: str,
    dpath: str,
    refpath: str,
    mpath: Optional[str],
    resolution: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
):
    """
    Generate images using pretrained network pickle.
    """
    
    '''random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)'''

    print(f'Loading data from: {dpath}')
    img_list = sorted(glob.glob(dpath + '/*.png') + glob.glob(dpath + '/*.jpg') + glob.glob(dpath + '/*.JPG'))

    if refpath is not None:
        print(f'Loading mask from: {mpath}')
        ref_list = sorted(glob.glob(refpath + '/*.png') + glob.glob(refpath + '/*.jpg') + glob.glob(refpath + '/*.JPG'))

    if mpath is not None:
        print(f'Loading mask from: {mpath}')
        mask_list = sorted(glob.glob(mpath + '/*.png') + glob.glob(mpath + '/*.jpg') + glob.glob(mpath + '/*.JPG'))
        print(img_list)
        print("#####")
        print(mask_list)
        assert len(img_list) == len(mask_list), 'illegal mapping'

    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cuda')
    #with open(network_pkl, 'rb') as f:
    with dnnlib.util.open_url(network_pkl) as f:
        #G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
        G_saved = pickle.load(f).to(device).eval().requires_grad_(False)
    f.close()
    #G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=resolution, img_channels=3).to(device).eval().requires_grad_(False)
    G = G_saved
    #copy_params_and_buffers(G_saved, G, require_all=True)

    os.makedirs(outdir, exist_ok=True)
    im_mask_pair = []
    # no Labels.
    label = torch.zeros([100, G.c_dim], device=device)

    def read_image(image_path):
        with open(image_path, 'rb') as f:
            if pyspng is not None and image_path.endswith('.png'):
                image = pyspng.load(f.read())
                print(image.shape)
            else:
                image = np.array(PIL.Image.open(f))
                print(image.shape)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
            image = np.repeat(image, 3, axis=2)
        if image.shape[1] != 512:
            image = rescale(image, 0.5)
        image = image.transpose(2, 0, 1) # HWC => CHW
        image = image[:3]
        print("image shape", image.shape)
        return image
     



    def convert_pixel_range(img, src, dst):
        if src != dst:
            src, dst = np.float32(src), np.float32(dst)
        img = np.clip(img, src[0], src[1])
        scale = (dst[1] - dst[0]) / (src[1] - src[0])
        bias = dst[0] - src[0] * scale
        img = img * scale + bias
        return img


	
    def to_opencv_image(img, in_range=None):
        if in_range is None:
            in_range = [-1, 1]
        img = convert_pixel_range(img.squeeze(), in_range, [0, 255])
        img = np.uint8(np.round(to_np(torch.permute(img, (1, 2, 0)))))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img









    def blend(src_img: torch.Tensor, dst_img: torch.Tensor, mask: torch.Tensor):
        src_img = to_opencv_image(src_img)
        dst_img = to_opencv_image(dst_img)
        mask = 1 - (mask)  # Invert mask, take hidden region from src instead of hiding it
        mask_np = np.uint8(255 * mask.squeeze().cpu())
        kernel = np.ones((20, 20), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        br = cv2.boundingRect(mask_np)  # bounding rect (x,y,width,height)
        center = (br[0] + br[2] // 2, br[1] + br[3] // 2)
        out = cv2.seamlessClone(dst_img, src_img, mask_np, center, cv2.NORMAL_CLONE)
        out = to_torch_image(out)
        return out



    def to_np(x):
        return x.cpu().detach().double().numpy()



    def to_torch_image(img, out_range=None):
        if out_range is None:
            out_range = [-1, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
        img = convert_pixel_range(img, [0, 255], out_range)
        return img






    def to_image(image, lo, hi):
        image = np.asarray(image, dtype=np.float32)
        image = (image - lo) * (255 / (hi - lo))
        image = np.rint(image).clip(0, 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image

    if resolution != 512:
        noise_mode = 'random'
    
    '''with torch.no_grad():
        for i, ipath in enumerate(ref_list):
            iname = os.path.basename(ipath).replace('.jpg', '.png')
            print(f'Processing: {iname}')
            ref_image = read_image(ipath)
            print("ref shape", ref_image.shape)
            ref_im_f = (torch.from_numpy(ref_image).float().to(device) / 127.5 - 1).unsqueeze(0)
            ref_image = (torch.from_numpy(ref_image).float().to(device) / 127.5 - 1).unsqueeze(0)'''
    
    with torch.no_grad():
        for i, ipath in enumerate(ref_list):
            iname = os.path.basename(ipath).replace('.jpg', '.png')
            print(f'Processing: {iname}')
            ref_image = read_image(ipath)
            #w_ref = calc_inversions(G, ref_image, 'ref')
            print("ref shape", ref_image.shape)
            ref_im_f = (torch.from_numpy(ref_image).float().to(device) / 127.5 - 1).unsqueeze(0)
            ref_image = (torch.from_numpy(ref_image).float().to(device) / 127.5 - 1).unsqueeze(0)
    #w_ref = calc_inversions(G, ref_im_f, 'ref')
    with torch.no_grad():
        for i, ipath in enumerate(img_list):
            iname = os.path.basename(ipath).replace('.jpg', '.png')
            print(f'Prcessing: {iname}')
            image = read_image(ipath)
            #w_image = calc_inversions(G, image, 'target')
            imfeat = (torch.from_numpy(image).float().to(device) / 127.5 - 1).unsqueeze(0)
            image = (torch.from_numpy(image).float().to(device) / 127.5 - 1).unsqueeze(0).view(1, 3, 512, 512)
            #w_image = calc_inversions(G, imfeat, 'target')

            if mpath is not None:
                mask = cv2.imread(mask_list[i], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                mask = cv2.resize(mask, dsize=(512,512))
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0).view(1, 1, 512, 512)
                mask = (mask * -1) + 1
            else:
                mask = RandomMask(resolution) # adjust the masking ratio by using 'hole_range'
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)
            im_mask_pair.append((iname, image, mask))
    #w_image = calc_inversions(G, imfeat, 'target')

    '''z_samples = np.random.randn(100, G.z_dim)
    print("kkkkk")
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c=None)  # [N, L, C]
    print("111")
    #w_samples = torch.load('./w.pt').detach()
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    #print(w_samples)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / 1) ** 0.5
    print("w_std", w_std)
    #w_avg = w_samples
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    #print("#####", w_opt.shape)
    w_out = torch.zeros([40] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=0.1)
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True'''


    ls = IDLoss()
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)
    target_images = image.unsqueeze(0).to(device).to(torch.float32)
    if image.shape[2] > 256:
        imagell = (image + 1) / 2
        imagell = F.interpolate(imagell, size=(256, 256), mode='area')
    #print(imagell)
    target_features = vgg16(imagell, resize_images=False, return_lpips=True)
    target_facenet = ls.extract_feats((image + 1 ) / 2)
    #print("??", torch.min(ref_image))
    #print("??", torch.max(ref_image))
    #print(ref_image)
    #ref_image = torchvision.transforms.GaussianBlur(kernel_size=(9, 9), sigma=(14, 14))(ref_image)
    ref_facenet_feats = ls.extract_feats(ref_image)
    if ref_image.shape[2] > 256:
        #ref_facenet_feats = ls.extract_feats(ref_image)
        #refim = (ref_image + 1) * 127.5
        refim = F.interpolate(ref_image, size=(256, 256), mode='area')
        print(refim)
    ref_features = vgg16(F.interpolate(refim, size=(256, 256), mode='area'), resize_images=False, return_lpips=True)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    w_std = 1
    im_num = 0
    lpips_loss = lpips.LPIPS(net='vgg').to(device)

    ref_facenet_feats = ls.extract_feats(ref_image)
    original_D = load_old_D()
    lpips_sim = []
    for name, image, mask in im_mask_pair:

        z_samples = np.random.randn(1, G.z_dim)
        #print("kkkkk")
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c=None)  # [N, L, C]
        #print("111")
        #w_samples = torch.load('./w.pt').detach()
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
        #print(w_samples)
        w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
        w_std = (np.sum((w_samples - w_avg) ** 2) / 1) ** 0.5
        #print("w_std", w_std)
        #w_avg = w_samples
        noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

        w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
        #print("#####", w_opt.shape)
        w_out = torch.zeros([1] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=0.1)
        
        for step in range(1):

            t = step / 1
            #print(step)
            w_noise_scale = w_std * 0.1 * max(0.0, 1.0 - t / 0.55) ** 2
            lr_ramp = min(1.0, (1.0 - t) / 0.25)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / 0.05)
            lr = 0.02 * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            w_noise = torch.randn_like(w_opt) * w_noise_scale 
            ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
            output, im = G.synthesis(image, mask, ws, noise_mode='const', return_stg1=True)
            x = image * mask + im * (1 - mask)
            with torch.no_grad():
                c = torch.empty([1, G.c_dim], device=device)
                gen_logits, gen_logits_stg1 = original_D(output, mask, im, c)
            loss_d = torch.nn.functional.softplus(-gen_logits)







        
            if step % 1 == 0:
                outputf = (im.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                #im_to_save = PIL.Image.fromarray(outputf[0].cpu().numpy(), 'RGB')
                

                outputg = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                im_to_save1 = PIL.Image.fromarray(outputg[0].cpu().numpy(), 'RGB')

                #equalized_im = ImageOps.equalize(im_to_save, mask=None)
                lpips_loss_calc = lpips_loss(image, output)
                #print(lpips_loss_calc)
                lpips_sim.append(lpips_loss_calc.cpu().detach().numpy()[0])
                #im_to_save.save(f'{outdir}/im_x_' + name.split('.')[0] + '_' + str(step) + '.png')

                im_to_save1.save(f'{outdir}/im_out_' + name.split('.')[0]+ '_'  + str(step)+ '.png')
                #im_to_save.save(f'{outdir}/im{step:02d}.png')
                #im = preprocess(Image.open(f'{outdir}/im{step:02d}.png')).unsqueeze(0).to(device)
                #print(image)
                #text = clip.tokenize(["man teeth"]).to(device)
                #image_features = clip_model.encode_image(output)
                #text_features = clip_model.encode_text(text)
                #PIL.Image.fromarray(outputstg1[0].cpu().numpy(), 'RGB').save(f'{outdir}/stg1{step:02d}.png')
                '''for i in [0.5, 2, 5, 10, 20, 30, 50, 100, 200, 300, 500]:
                direction_to_move = i * interpolation_direction / interpolation_direction_norm
                result_w = ws + (i * w_randdir)#direction_to_move
                output = G.synthesis(image, mask, result_w, noise_mode='const')
                outputf = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(outputf[0].cpu().numpy(), 'RGB').save(f'{outdir}/im{step + i:04f}.png')'''
        
            synth_images = output 

            cim = PIL.Image.fromarray((outputf[0].cpu().numpy() + 1) * 127.5, 'RGB')
            #if synth_images.shape[2] > 256:
            #print("!!", torch.min(synth_images))
            #print("!!", torch.max(synth_images))
            l2_loss_val = l2_loss.l2_loss((synth_images + 1) *127.5 , (ref_image + 1) * 127.5)
        

            synth_features = vgg16(F.interpolate(synth_images ,size=(256, 256), mode='area'), resize_images=False, return_lpips=True)
            synth_images = torchvision.transforms.GaussianBlur(kernel_size=(9, 9), sigma=(14, 14))(synth_images)
            facenet_synth_features = ls.extract_feats(synth_images)
            dist = (ref_features - synth_features).square().sum()

            #output = G(image, mask, z, label, truncation_psi=truncation_psi, noise_mode='none')
            reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            #cimage = preprocess(cim).unsqueeze(0).to(device)
        
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            #loss = (1 - cos(image_features, text_features))
            loss = (1 - (cos(facenet_synth_features, ref_facenet_feats)))  #+ reg_loss * (1e5)
            print(cos(facenet_synth_features, ref_facenet_feats))
            #print(ref_image)
            #loss = (facenet_synth_features - ref_facenet_feats).square().sum()
            #print(loss)
            #loss = dist
            #loss = l2_loss_val
            #print(loss)
            optimizer.zero_grad()
            #print("#####", id_los_calc, "#####")
            (loss).backward(retain_graph=True)
            optimizer.step()
            im_num +=1
        #lpips_loss_calc = lpips_loss(image, synth_images)
        #print(lpips_loss_calc)
        #lpips_sim.append(lpips_loss_calc.cpu().detach().numpy()[0])
    print('std', np.std(lpips_sim))
    print('mean', np.mean(lpips_sim))



if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
