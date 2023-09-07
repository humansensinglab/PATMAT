import torch
import numpy as np
import wandb
import cv2
import random
import PIL.Image
import glob
from criteria import l2_loss
from configs import hyperparameters
from configs import global_config
from loss import IDLoss
import random
import torchvision
import os
import pyspng
from utils.models_utils import toogle_grad, load_old_G
mpath = './segmentation'
#mask_list = sorted(glob.glob(mpath + '/*.png') + glob.glob(mpath + '/*.jpg') + glob.glob(mpath + '/*.JPG'))
#mask = cv2.imread('./segmentation/3.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
#print("mask shape",  mask.shape)
#mask = cv2.resize(mask, dsize=(512,512))

mask = torch.from_numpy(mask).float().to(global_config.device).unsqueeze(0).view(1, 1, 512, 512)

#print("mask shape", mask.shape)
mask = (mask * -1) + 1
ls = IDLoss()

##################################### replace with your own regularizing data ##############################
img_list = sorted(glob.glob('celeba' + '/*.png') + glob.glob('celeba' + '/*.jpg') + glob.glob('celeba' + '/*.JPG'))


oldG = load_old_G()


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
    #print("image shape", image.shape)
    image = torch.from_numpy(image).float().to(global_config.device)
    #print("!!!!!!!!!!!!!!!!", image)
    return (image / 127.5) - 1

def to_image(image, lo, hi):
    image = np.asarray(image, dtype=np.float32)
    image = (image - lo) * (255 / (hi - lo))
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return image

class Space_Regulizer:
    def __init__(self, original_G, lpips_net):
        self.original_G = original_G
        self.morphing_regulizer_alpha = hyperparameters.regulizer_alpha
        self.lpips_loss = lpips_net

    def get_morphed_w_code(self, target_w_code, fixed_w):
        interpolation_direction = target_w_code - fixed_w
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
        direction_to_move = 40 * interpolation_direction / interpolation_direction_norm

        result_w = fixed_w + direction_to_move

        return result_w

    def get_image_from_ws(self, image, w_codes, G):
        return torch.cat([G.synthesis(image, mask, w_code, noise_mode='none') for w_code in w_codes])

    def ball_holder_loss_lazy(self, celeb_images, celeb_pivots, w_pivots, target_images, target_w, image, new_G, num_of_sampled_latents, w_batch, use_wandb=False):
        loss = 0.0
        z_samples = np.random.randn(1, new_G.z_dim)
        w_randdir = self.original_G.mapping(torch.from_numpy(z_samples).to(global_config.device), None,
                                            truncation_psi=0.5)
        
        w_0 = w_pivots[0]
        for i in range(1, len(w_pivots)):
            w_0 = torch.vstack((w_0.view(-1, 12, 512), w_pivots[i]))

        #territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_randdir]
        #territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]
    
        for w_code in [w_batch]:
            t = 0
            #new_img = new_G.synthesis(image, mask, w_code)
            #new_img_fix = new_G.synthesis(image, mask, w_batch)
            #outputf = (new_img.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            #PIL.Image.fromarray(outputf[0].cpu().numpy(), 'RGB').save('localityreg/im' + '.png')
            with torch.no_grad():    
                for data, w in zip(celeb_images, celeb_pivots):
                    image_name, image = data
                    
                    
                    old_img, oim = oldG.synthesis(image, mask, w, return_stg1=True)
                    outputf = (old_img.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                    #PIL.Image.fromarray(outputf[0].cpu().numpy(), 'RGB').save('origs/im'+image_name + "_" + str(t) + '.png')
                    with torch.enable_grad():
                        new_img, img = new_G.synthesis(image, mask, w, return_stg1=True)
                        outputf = (new_img.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                        PIL.Image.fromarray(outputf[0].cpu().numpy(), 'RGB').save('regs/im' + image_name + "_" +  str(t) + '.png')
            
                    if hyperparameters.regulizer_l2_lambda > 0:
                        l2_loss_val = l2_loss.l2_loss(image, new_img)
                        old = ls.extract_feats((image + 1) / 2)
                        new = ls.extract_feats((new_img + 1) / 2)
                        if use_wandb:
                            wandb.log({f'space_regulizer_l2_loss_val': l2_loss_val.detach().cpu()},
                              step=global_config.training_step)
                        

                        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                        id_loss = (1 - cos(old, new))[0] * 1
                        print("###########", id_loss)
                        loss += id_loss #* hyperparameters.regulizer_l2_lambda
                        t += 1
                        #loss += id_loss
                    if hyperparameters.regulizer_lpips_lambda > 0:
                        loss_lpips = self.lpips_loss(image, new_img)
                        loss_lpips = torch.mean(torch.squeeze(loss_lpips))
                        if use_wandb:
                            wandb.log({f'space_regulizer_lpips_loss_val': loss_lpips.detach().cpu()},
                              step=global_config.training_step)

                        loss += loss_lpips * hyperparameters.regulizer_lpips_lambda
            
            #styleGAN propert regularization
            stylegan_reg =  False
            if stylegan_reg == True:
                lpips_old = self.lpips_loss((old_img_fix - old_img), (new_img_fix - new_img))
                lpips_loss_stylegan = torch.mean(torch.squeeze(lpips_old))
                #lpips_new = self.lpips_loss((new_img_fix - new_img))
                #style_property_loss = torch.abs(torch.mean(torch.squeeze(lpips_old)) - torch.mean(torch.squeeze(lpips_new)))
                #cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                loss +=  lpips_loss_stylegan * 1

            '''for target_image in target_images:
                weighted_sum_coef = torch.randn(len(w_pivots)).to(global_config.device)
                #print(w_0.shape)
                #w_weighted_sum = res = torch.einsum('mbchw,m->bcw', weighted_sum_coef, w_0)
                #print("########################")
            
                w_av = w_0[0] * random.uniform(1, 3)
                for i in range(1, w_0.shape[0]):
                    #print(w_0[i].shape)

                    w_av += w_0[i] * random.uniform(1, 3) #weighted_sum_coef[i]
                
                
                #print(w_av.shape)

                #cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)'''
        

                ##print("imageeeeee", image)
                ##target_image = torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=(5,7))(target_image)
                #weighted_im = new_G.synthesis(target_image, mask, w_av.view(1, 12, 512))
                ##weighted_im = torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=(5,7))(weighted_im)
                #target_feat = ls.extract_feats((weighted_im + 1) / 2)
                ##image = torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=(5,7))(image)
                #tuning_feat = ls.extract_feats(image)
                ##print("loss for cosine sim",  (1 - cos(target_feat, tuning_feat))[0])
                #loss += (1 - cos(target_feat, tuning_feat))[0] * 1
       
        #print("this is the regularization loss", loss / len(celeb_images))
        return loss  / len(celeb_images)

    def space_regulizer_loss(self, celeb_images, celeb_pivots, w_pivots, target_image, target_w, image, new_G, w_batch, use_wandb):
        ret_val = self.ball_holder_loss_lazy(celeb_images, celeb_pivots, w_pivots, target_image, target_w, image, new_G, hyperparameters.latent_ball_num_of_samples, w_batch, use_wandb)
        return ret_val
