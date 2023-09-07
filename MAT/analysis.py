import cv2
#from criteria import l2_loss
import pyspng
import imageio
import glob
import os
import lpips
import itertools

#ifrom projectors import w_projector
#from utils.models_utils import toogle_grad, load_old_G, load_old_D
#from tqdm import tqdm
import re
import random
from loss import IDLoss
#from networks.basic_module import FullyConnectedLayer, Conv2dLayer, MappingNet
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

device = torch.device('cuda')


dpath = './patmat_quant/ts/gt'
tpath = './forfid/paintmyts/out_image'

gt_list = sorted(glob.glob(dpath + '/*.png') + glob.glob(dpath + '/*.jpg') + glob.glob(dpath + '/*.JPG'))
targ_list = sorted(glob.glob(tpath + '/*.png') + glob.glob(tpath + '/*.jpg') + glob.glob(tpath + '/*.JPG'))

def read_image(image_path):
    with open(image_path, 'rb') as f:
        if pyspng is not None and image_path.endswith('.png'):
            image = pyspng.load(f.read())
        else:
            image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
            image = rescale(image, 0.5)
        image = image.transpose(2, 0, 1)
        image = image[:3]
        return image

ls = IDLoss().to(device)
lpips_loss = lpips.LPIPS(net='vgg').to(device)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
gt_ims = []
tar_ims = []
assert len(gt_ims) == len(tar_ims)
for i, ipath in enumerate(gt_list):
    im = read_image(ipath)
    gt_ims.append((torch.from_numpy(im).float().to(device) / 127.5 - 1).unsqueeze(0))

lpips_res = []
id_res = []

for i, ipath in enumerate(targ_list):
    l_loss = 0
    id_sim_sum = 0
    with torch.no_grad():
        im = read_image(ipath) / 127.5 - 1
        im = torch.from_numpy(im).view(1, 3, 1024, 1024).to(torch.float).to(device)
        print(im.shape)
        tt = ls.extract_feats(im)
        for im, src in list(itertools.product([im], gt_ims)):
            #print(im)
            #print(src)
            #l_loss += lpips_loss(im, src)
            src = src.view(1, 3, 512, 512).to(torch.float).to(device)
            sr = ls.extract_feats(src)
            id_sim = cos(sr, tt).cpu().numpy()[0]
            #print(l_loss)
            print(id_sim)
            id_sim_sum += id_sim
        id_res.append(id_sim_sum / len(gt_ims))
print('mean is', np.mean(id_res))
print('std is', np.std(id_res))
















