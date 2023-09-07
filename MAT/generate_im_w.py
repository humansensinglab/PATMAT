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
from utils.models_utils import toogle_grad, load_old_G
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
import pickle
import numpy as np
from skimage.transform import resize, rescale
import PIL.Image
import torchvision
import torch
import torch.nn.functional as F
from torch import optim
import legacy
#from datasets.mask_generator_512 import RandomMask
from networks.mat import Generator

device = torch.device('cuda')


def calc_inversions(G, image, image_name):
        id_image = torch.squeeze((image + 1) / 2) * 255
        w = w_projector.project(G, id_image, device=device, w_avg_samples=600,
                                    num_steps=150, w_name=image_name,
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
    '''seed = 426  # pick up a random number
    random.seed(seed)
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
        print(ref_list)
        print("#####")
        print(mask_list)
        assert len(img_list) == len(mask_list), 'illegal mapping'

    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cuda')
    #with open(network_pkl, 'rb') as f:
    with dnnlib.util.open_url(network_pkl) as f:
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
        #G_saved = pickle.load(f).to(device).eval().requires_grad_(False)
    f.close()
    #G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=resolution, img_channels=3).to(device).eval().requires_grad_(False)
    G = G_saved
    #copy_params_and_buffers(G_saved, G, require_all=True)

    os.makedirs(outdir, exist_ok=True)

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
    #w_image = calc_inversions(G, imfeat, 'target')

    z_samples = np.random.RandomState(1).randn(10, G.z_dim)
    #print("kkkkk", G.mapping.num_ws)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    #print("111", w_samples.shape)
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    #print(w_samples)











    '''z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
    output = G(image, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
    output = output[0].cpu().numpy()
    PIL.Image.fromarray(output, 'RGB').save('samples/' + str(i)+ '.png')'''
    for i in range(20):
        z = np.random.randn(1, G.z_dim)
        print(z.shape)
        w_samples = G.mapping(torch.from_numpy(z).to(device), None)  # [N, L, C]
        z_p = np.random.randn(1, G.z_dim)
        w_p = G.mapping(torch.from_numpy(z).to(device), None)


        z_q = np.random.randn(1, G.z_dim)
        w_q = G.mapping(torch.from_numpy(z).to(device), None)
        #w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
        #w_samples = np.mean(w_samples, axis=0, keepdims=True)
        #w = w_samples.repeat([1, 12, 1])
        #print("111", w_samples.shape)
        #w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
        w = 0.3 * w_samples + 0.4 * w_q + 0.3 * w_p
        output = G.synthesis(image, mask, w + w_p, noise_mode=noise_mode)
        output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        output = output[0].cpu().numpy()
        PIL.Image.fromarray(output, 'RGB').save('gen/' + str(i)+ '.png')
    '''with torch.no_grad():
        w_samples = torch.load('./start_w.pt') #/ 255
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
        print("eeeeeee", w_samples)
        #print("hhhhhhhhhhhhh", w_samples.shape)'''
    

    '''w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / 1) ** 0.5
    print("w_std", w_std)
    #w_avg = w_samples
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    #print("#####", w_opt.shape)

    #print(imagell)
    target_features = vgg16(imagell, resize_images=False, return_lpips=True)
    target_facenet = ls.extract_feats(image)
    print("??", torch.min(image))
    print("??", torch.max(image))
    print(ref_image)
    ref_image = torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=(5, 7))(ref_image)
    ref_facenet_feats = ls.extract_feats((ref_image + 1) / 2)
    if ref_image.shape[2] > 256:
        #ref_facenet_feats = ls.extract_feats(ref_image)
        #refim = (ref_image + 1) * 127.5
        refim = F.interpolate(ref_image, size=(256, 256), mode='area')
        print(refim)
    ref_features = vgg16(F.interpolate(refim, size=(256, 256), mode='area'), resize_images=False, return_lpips=True)
    output = G.synthesis(image, mask, w_image, noise_mode='const')
    output2 = G.synthesis(ref_image, mask, w_ref, noise_mode='const')
    
    output_rev = G.synthesis(image, mask, w_ref, noise_mode='const')
    output2_rev = G.synthesis(ref_image, mask, w_image, noise_mode='const')
    

    outputim2 = (output_rev.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
    outputref2 = (output2_rev.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
    
    outputim = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
    outputref = (output2.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)

    PIL.Image.fromarray(outputim[0].cpu().numpy(), 'RGB').save('samples/outim.png')
    PIL.Image.fromarray(outputref[0].cpu().numpy(), 'RGB').save('samples/outref.png')
    PIL.Image.fromarray(outputim2[0].cpu().numpy(), 'RGB').save('samples/outimrev.png')
    PIL.Image.fromarray(outputref2[0].cpu().numpy(), 'RGB').save('samples/outrefrev.png')
    i = 0
    for c in np.linspace(0, 1, 50):
        output = G.synthesis(ref_image, mask, (1 - c) * w_image + (c) * w_ref)
        output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(output[0].cpu().numpy(), 'RGB').save('samples/' + str(i) + 'outim.png')
        i += 1
        synth_images = output 
        cim = PIL.Image.fromarray((outputf[0].cpu().numpy() + 1) * 127.5, 'RGB')
        #if synth_images.shape[2] > 256:
        print("!!", torch.min(output))
        print("!!", torch.max(output))
        l2_loss_val = l2_loss.l2_loss((synth_images + 1) *127.5 , (ref_image + 1) * 127.5)
        synth_features = vgg16(F.interpolate((synth_images + 1) / 2 ,size=(256, 256), mode='area'), resize_images=False, return_lpips=True)
        synth_images = torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=(5, 7))(synth_images)
        facenet_synth_features = ls.extract_feats((synth_images + 1) / 2)
        dist = (target_features - synth_features).square().sum()

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
        cimage = preprocess(cim).unsqueeze(0).to(device)
        

        #loss =  d
        #similarity = (image_features @ text_features.T)
        #similarity.requires_grad=True
        #print(similarity[0][0])
        #with torch.no_grad():
        
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        #loss = (1 - (cos(facenet_synth_features, ref_facenet_feats))) #+ reg_loss * 1e5
        #print(loss)
        loss = dist
        #loss = l2_loss_val
        print("loss", loss)
        optimizer.zero_grad()
        #print("#####", id_los_calc, "#####")
        loss.backward(retain_graph=True)
        optimizer.step()
        w_out[step] = w_opt.detach()[0]

    projected_w_steps = w_out.repeat([1, G.mapping.num_ws, 1])
    
    video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
    print (f'Saving optimization progress video "{outdir}/proj.mp4"')
    
    for projected_w in projected_w_steps[:]:
        synth_image = G.synthesis(image, mask,projected_w.unsqueeze(0), noise_mode='random')
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        video.append_data(np.concatenate([synth_image], axis=1))
    video.close()


    #projected_w = projected_w_steps[-1]
    
    #output = G(image, mask, projected_w.unsqueeze(0), label, noise_mode=noise_mode)
    #output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
    #output = output[0].cpu().numpy()
    
    #PIL.Image.fromarray(output, 'RGB').save(f'{outdir}/{iname}')'''


if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
