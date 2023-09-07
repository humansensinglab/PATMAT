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
import lpips
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
import statistics 

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



def RandomMask(s, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((s, s), np.uint8)
        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)

            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
        def MultiFill(max_tries, max_size):

            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)
        MultiFill(int(5 * coef), s // 2)
        MultiFill(int(3 * coef), s)
        mask = np.logical_and(mask, 1 - RandomBrush(int(9 * coef), s))  # hole denoted as 0, reserved as 1
        hole_ratio = 1 - np.mean(mask)

        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return mask[np.newaxis, ...].astype(np.float32)







def RandomBrush(
    max_tries,
    s,
    min_num_vertex = 4,
    max_num_vertex = 18,
    mean_angle = 2*math.pi / 5,
    angle_range = 2*math.pi / 15,
    min_width = 12,
    max_width = 48):
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask




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
@click.option('--tpath', help='the path of the training image', required=True)
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
    tpath: str,
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
    train_list = sorted(glob.glob(tpath + '/*.png') + glob.glob(tpath + '/*.jpg') + glob.glob(tpath + '/*.JPG'))
    if refpath is not None:
        print(f'Loading mask from: {mpath}')
        ref_list = sorted(glob.glob(refpath + '/*.png') + glob.glob(refpath + '/*.jpg') + glob.glob(refpath + '/*.JPG'))

    if mpath is not None:
        print(f'Loading mask from: {mpath}')
        mask_list = sorted(glob.glob(mpath + '/*.png') + glob.glob(mpath + '/*.jpg') + glob.glob(mpath + '/*.JPG'))
        print(img_list)
        print("#####")
        print(mask_list)
        
    
    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cuda')
    with open(network_pkl, 'rb') as f:
    #with dnnlib.util.open_url(network_pkl) as f:
        #G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
        G_saved = pickle.load(f).to(device).eval().requires_grad_(False)
    f.close()
    #G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=resolution, img_channels=3).to(device).eval().requires_grad_(False)
    G = G_saved
    #copy_params_and_buffers(G_saved, G, require_all=True)

    os.makedirs(outdir, exist_ok=True)
    im_mask_pair = []
    # no Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    def read_image(image_path):
        with open(image_path, 'rb') as f:
            if pyspng is not None and image_path.endswith('.png'):
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
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

    train_ims = []

    with torch.no_grad():
        for i, ipath in enumerate(train_list):
            iname = os.path.basename(ipath).replace('.jpg', '.png')
            train_image = read_image(ipath)
            train_image = (torch.from_numpy(train_image).float().to(device) / 127.5 - 1).unsqueeze(0).view(1, 3, 512, 512)
            train_ims.append(train_image)

    #w_ref = calc_inversions(G, ref_im_f, 'ref')
    with torch.no_grad():
        for i, ipath in enumerate(img_list):
            iname = os.path.basename(ipath).replace('.jpg', '.png')
            #print(f'Prcessing: {iname}')
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
            im_mask_pair.append((image, mask))
    #w_image = calc_inversions(G, imfeat, 'target')

    ls = IDLoss()
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    
    w_std = 1
    im_num = 0
    id_sim = []
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    ref_facenet_feats = ls.extract_feats(ref_image)
    original_D = load_old_D()
    with torch.no_grad():
        i = 0
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        for image, _ in im_mask_pair:
            #print(mask.shape)
            #msk = RandomMask(512)
            #mask = torch.from_numpy(msk).float().to(device).unsqueeze(0)
            #print(mask.shape)
            #z_samples = np.random.randn(100, G.z_dim)
            #print("kkkkk")
            #w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c=None)  # [N, L, C]
            #print("111")
            #w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
            #print(w_samples)
            #w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
            #w_std = (np.sum((w_samples - w_avg) ** 2) / 1) ** 0.5
            #print("w_std", w_std)
            #w_avg = w_samples
            #noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

            #w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=False) # pylint: disable=not-callable
            #print("#####", w_opt.shape)
            #w_out = torch.zeros([80] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
        
        
            for step in range(5):
                im_num = 0
                t = step / 5
                msk = RandomMask(512)
                mask = torch.from_numpy(msk).float().to(device).unsqueeze(0)
                #print(step)
        
                z_samples = np.random.randn(1, G.z_dim)
                #print('z', z_samples.shape)
                w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c=None)
                w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
                #print('w_sam', w_samples.shape)
                w_avg = np.mean(w_samples, axis=0, keepdims=True)
                #print('before', w_samples.shape)
                ws = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=False)
                ws = ws.repeat([1, G.mapping.num_ws, 1])
                #print('after', ws.shape)

                #ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
                output, im = G.synthesis(image, mask, ws, noise_mode='const', return_stg1=True)


                #imf = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                #imf = imf[0].cpu().numpy()
                #imf = PIL.Image.fromarray(imf, 'RGB')
                #msk = Image.fromarray((msk * 255).astype(np.uint8))
                #msks = PIL.Image.fromarray((msk[0] * 255).astype(np.uint8)).convert('L')
                outputf = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                #outputf = outputf[0].cpu().numpy()
                #outputf = PIL.Image.fromarray(outputf, 'RGB')
                #mas = Image.fromarray(maskf[0].cpu().numpy())
                #print(msk)
                '''msks = np.asarray(msks)
                #print(msk.shape)
                #print("........", msks.shape)
                br = cv2.boundingRect(msks) # bounding rect (x,y,width,height)
                centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)
                outputf = cv2.cvtColor(np.asarray(outputf), cv2.COLOR_BGR2RGB)
                imf = cv2.cvtColor(np.asarray(imf), cv2.COLOR_BGR2RGB)
                poissonImage = cv2.seamlessClone(np.asarray(outputf), np.asarray(imf), msks, centerOfBR, 0)
                print(poissonImage)
                cv2.imwrite('./try/' + str(step) + '.png', poissonImage)
                blended = (np.asarray(poissonImage).reshape((3, 512, 512))) / 127.5 - 1'''
                
            
                
                #print("this is blended shape",blended.shape)
                #print("this is the center", centerOfBR)
                im_to_save = PIL.Image.fromarray(outputf[0].cpu().numpy(), 'RGB')
                im_to_save.save(f'{outdir}/im{step:02d}' + str(i) + '_' +  str(step) + '.png')
                #im_num +=1
                synth_images = output
                #print("this is output shape that works", output.shape)
                facenet_synth_features = ls.extract_feats(synth_images)
                

                train_feats = ls.extract_feats(image)
                #blended_f = ls.extract_feats(blended)
                #cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                #loss = (1 - cos(image_features, text_features))
                #sim = cos(train_feats, blended_f)
                d = loss_fn_vgg(image, synth_images)
                id_sim.append(d.cpu().detach().numpy()[0])
    
                print(d)
            i +=1 
    arr = np.array(id_sim)

    print('std', np.std(id_sim))
    print('mean', np.mean(id_sim))

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
