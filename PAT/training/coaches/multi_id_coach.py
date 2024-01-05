import os
import itertools
import torch
from tqdm import tqdm
import pickle
from PIL import Image, ImageDraw
import PIL.Image
import numpy as np
import torchvision
import copy
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
import cv2
import time
import math
import random
#from datasets.mask_generator_512 import RandomMask
from utils.models_utils import toogle_grad, load_old_G, load_old_D
#mpath = './segmentation'
#mask_list = sorted(glob.glob(mpath + '/*.png') + glob.glob(mpath + '/*.jpg') + glob.glob(mpath + '/*.JPG'))
#mask = cv2.imread('./segmentation/3.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
#print("mask shape",  mask.shape)
mask = cv2.resize(mask, dsize=(512,512))

mask = torch.from_numpy(mask).float().to(global_config.device).unsqueeze(0).view(1, 1, 512, 512)

#print("mask shape", mask.shape)
mask = (mask * -1) + 1

original_G = load_old_G()
original_D = load_old_D()
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

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def BatchRandomMask(batch_size, s, hole_range=[0, 1]):
    return np.stack([RandomMask(s, hole_range=hole_range) for _ in range(batch_size)], axis=0)
toogle_grad(original_G, False)
class MultiIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self):
        self.G.synthesis.train()
        self.G.mapping.train()

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True
        w_pivots = []
        w_glasses = []
        w_p3 = []
        w_sunglasses = []
        w_p1 = []
        w_p2 = []
        w_p4 = []
        celeb_pivots = []



        images = []
        images_glasses = []
        images_sunglasses = []
        images_p1 = []
        images_p2 = []
        images_p3 = []
        images_p4 = []
        celeb_images = []
        
        for fname, image in self.data_loader:
            print(fname[0])
        print("#####")
        for fname, image in self.data_loader:
            if self.image_counter >= hyperparameters.max_images_to_invert:
                break
            image_name = fname[0]
            pivot_ws = []
            target_images = []
            print(image_name)
            if 'target' in image_name:
                if hyperparameters.first_inv_type == 'w+':
                    embedding_dir = f'{w_path_dir}/{paths_config.e4e_results_keyword}/{image_name}'
                else:
                    embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
                os.makedirs(embedding_dir, exist_ok=True)
                #start_w = self.get_inversion(w_path_dir, image_name, image).to(global_config.device)
                target_image = image.to(global_config.device)
                start_w = None
                #torch.save(start_w, 'start_w.pt')
                pivot_ws.append(start_w)
                target_images.append(target_image)
            elif 'ffhq' in image_name:
                pass
            elif 'celeb' in image_name:
                if hyperparameters.first_inv_type == 'w+':
                    embedding_dir = f'{w_path_dir}/{paths_config.e4e_results_keyword}/{image_name}'
                else:
                    embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
                os.makedirs(embedding_dir, exist_ok=True)
                #start_w = self.get_inversion(w_path_dir, image_name, image).to(global_config.device)
                w_pivot = self.get_inversion(w_path_dir, image_name, image)
                #torch.save(start_w, 'start_w.pt')
                celeb_pivots.append(w_pivot.to(global_config.device))
                celeb_images.append((image_name, image.to(global_config.device)))
                
            else:
                #print("processing none target")
                if hyperparameters.first_inv_type == 'w+':
                    embedding_dir = f'{w_path_dir}/{paths_config.e4e_results_keyword}/{image_name}'
                else:
                    embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
                print(embedding_dir)
                os.makedirs(embedding_dir, exist_ok=True)
                #w_pivot = self.get_inversion(w_path_dir, image_name, image)
                
                w_pivot = self.get_inversion(w_path_dir, image_name, image)
                if 'g' in image_name:
                    w_glasses.append(w_pivot)
                    #image = torchvision.transforms.GaussianBlur(kernel_size = (7, 13), sigma=(9,11))(image)
                    images_glasses.append((image_name, image))
                
                elif 'p1' in image_name:
                    w_p1.append(w_pivot)
                    #image = torchvision.transforms.GaussianBlur(kernel_size = (7, 13), sigma=(9,11))(image)
                    images_p1.append((image_name, image))
                elif 'p2' in image_name:
                    
                    w_p2.append(w_pivot)
                    images_p2.append((image_name, image))
                elif 'p3' in image_name:
                    w_p3.append(w_pivot)
                    images_p3.append((image_name, image))
                elif 'p4' in image_name[:]:
                    w_p4.append(w_pivot)
                    images_p4.append((image_name, image))
                else:
                    pass
                    #w_p2.append(w_pivot)
                    #images_p2.append((image_name, image))
                self.image_counter += 1
        print("# sunglasses data", len(images_sunglasses))
        print("# p3 data", len(images_p3))
        print("# glasses data", len(images_glasses))
        print("# p1 data", len(images_p1))
        print("# p2 data", len(images_p2))
        print("# p4 data", len(images_p4))



        #build w subclusters for each category
        lam = 1


        cluster_p1 = []
        cluster_p2 = []
        cluster_p3 = []
        cluster_p4 = []
        cluster_glasses = []
        cluster_sunglasses = []



        for i in range(len(w_p1)):
            subcluster = [0] * len(w_p1)
            subcluster[i] = w_p1[i]
            for j in range(len(w_p1)):
                if j != i:
                    move_dir = w_p1[j] - w_p1[i]
                    with torch.no_grad():
                        dir_norm = torch.norm(move_dir, p=2)
                        #new_point = w_p1[i] + (move_dir / dir_norm)
                        new_point = torch.randn_like(w_p1[i]) + w_p1[i]
                        subcluster[j] = w_p1[i]
            cluster_p1.append(subcluster)
                
        for i in range(len(w_p2)):
            subcluster = [0] * len(w_p2)
            subcluster[i] = w_p2[i]
            for j in range(len(w_p2)):
                if j != i:
                    move_dir = w_p2[j] - w_p2[i]
                    with torch.no_grad():
                        dir_norm = torch.norm(move_dir, p=2)
                        #new_point = w_p2[i] + (move_dir / dir_norm)
                        new_point = torch.randn_like(w_p2[i]) + w_p2[i]
                        subcluster[j] =  w_p2[i]
            cluster_p2.append(subcluster)

        for i in range(len(w_p3)):
            subcluster = [0] * len(w_p3)
            subcluster[i] = w_p3[i]
            for j in range(len(w_p3)):
                if j != i:
                    move_dir = w_p3[j] - w_p3[i]
                    with torch.no_grad():
                        dir_norm = torch.norm(move_dir, p=2)
                        new_point_3 = w_p3[i] + (move_dir / dir_norm) 
                        new_point = torch.randn_like(w_p3[i]) + w_p3[i]
                        subcluster[j] = w_p3[i]
            cluster_p3.append(subcluster)


        for i in range(len(w_p4)):
            subcluster = [0] * len(w_p4)
            subcluster[i] = w_p4[i]
            for j in range(len(w_p4)):
                if j != i:
                    move_dir = w_p4[j] - w_p4[i]
                    with torch.no_grad():
                        dir_norm = torch.norm(move_dir, p=2)
                        new_point = w_p4[i] + (move_dir / dir_norm)
                        subcluster[j] =  w_p4[i]
            cluster_p4.append(subcluster)





        for i in range(len(w_glasses)):
            subcluster = [0] * len(w_glasses)
            subcluster[i] = w_glasses[i]
            for j in range(len(w_glasses)):
                if j != i:
                    move_dir = w_glasses[j] - w_glasses[i]
                    with torch.no_grad():
                        dir_norm = torch.norm(move_dir, p=2)
                        new_point_glasses = w_glasses[i] + (move_dir / dir_norm)
                        #new_point =  torch.randn_like(w_glasses[j]) + w_glasses[i] 
                        subcluster[j] = w_glasses[i]
            print(subcluster)
            cluster_glasses.append(subcluster)

        '''for w in w_p1:
            subcluster = []
            subcluster.append(w)
            for i in range(len(w_p1) - 1):
                w_noise = torch.randn_like(w)
                subcluster.append(w +  lam * w_noise)
            cluster_p1.append(subcluster)

        for w in w_p2:
            subcluster = []
            subcluster.append(w)
            for i in range(len(w_p2) - 1):
                w_noise = torch.randn_like(w)
                subcluster.append(w +  lam * w_noise)
            cluster_p2.append(subcluster)

        for w in w_p3:
            subcluster = []
            subcluster.append(w)
            for i in range(len(w_p3) - 1):
                w_noise = torch.randn_like(w)
                subcluster.append(w + lam * w_noise)
            cluster_p3.append(subcluster)

        for w in w_glasses:
            subcluster = []
            subcluster.append(w)
            for i in range(len(w_glasses) - 1):
                w_noise = torch.randn_like(w)
                subcluster.append(w +  lam * w_noise)
            cluster_glasses.append(subcluster)
        for w in w_sunglasses:
            subcluster = []
            subcluster.append(w)
            for i in range(len(w_sunglasses) - 1):
                w_noise = torch.randn_like(w)
                subcluster.append(w + lam * w_noise)
            cluster_sunglasses.append(subcluster)'''


        p1_pairs = []
        p2_pairs = []
        p3_pairs = []
        p4_pairs = []
        glasses_pairs = []
        sunglasses_pairs = []

        for subcluster in cluster_p1:
            p1_pairs += list(zip(images_p1, subcluster))

        for subcluster in cluster_p4:
            p4_pairs += list(zip(images_p4, subcluster))
        for subcluster in cluster_p2:
            p2_pairs += list(zip(images_p2, subcluster))
        for subcluster in cluster_p3:
            p3_pairs += list(zip(images_p3, subcluster))
        for subcluster in cluster_glasses:
            glasses_pairs += list(zip(images_glasses, subcluster))
        for subcluster in cluster_sunglasses:
            sunglasses_pairs += list(zip(images_sunglasses, subcluster))




        iter_list = p1_pairs + p2_pairs + p3_pairs + p4_pairs + glasses_pairs + sunglasses_pairs

        print("############### final number of pairs via sub-clusters ###########################")
        print(len(iter_list))
        


        #celeb_im_w_pair = list(itertools.product(celeb_images, celeb_pivots))
        celeb_im_w_pair = list(zip(celeb_images, celeb_pivots))
        start = time.time()
        for i in tqdm(range(hyperparameters.max_pti_steps)):
            self.image_counter = 0
            celeb_G = []
            celeb_im_to_pass = []
            j = 0
            #test all combinations of ws and images instead of actual pairs
            #for data, w_pivot in (list(itertools.product(images_p1,cluster_p1)) + list(itertools.product(images_p3,cluster_p3))+ list(itertools.product(images_glasses,cluster_glasses)) + list(itertools.product(images_sunglasses,w_sunglasses)) +list(itertools.product(images_p2,cluster_p2))):
            #for data, w_pivot in zip(images_p1,w_p1):# + list(zip(images_glasses, w_glasses)):
            for data, _ in iter_list:
                print(j)
                

                z_samples = np.random.randn(1, self.G.z_dim)
                w_samples = self.G.mapping(torch.from_numpy(z_samples).to(global_config.device), c=None)
                w_pivot = w_samples

                image_name, image = data
                celeb_rand_idx = random.randint(0, len(celeb_im_w_pair) - 1)

                
                w_celeb = celeb_pivots[celeb_rand_idx]
                
                image_to_use = celeb_images[celeb_rand_idx][1]
                mask = RandomMask(512) # adjust the masking ratio by using 'hole_range'
                mask = torch.from_numpy(mask).float().to(global_config.device).unsqueeze(0)
                data, w_celeb = celeb_im_w_pair[celeb_rand_idx]
                z_samples = np.random.randn(1, self.G.z_dim)
                w_samples = self.G.mapping(torch.from_numpy(z_samples).to(global_config.device), c=None)
                w_celeb = w_samples1
                celeb_name, image_to_use = data
                #print(next(self.G.parameters()).is_cuda)
                generated_celeb_images, imceleb = self.G.synthesis(image_to_use, mask, w_celeb, return_stg1=True)
                outputcel = (generated_celeb_images.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
              
                with torch.no_grad():
                    
                    #print(next(original_G.parameters()).is_cuda)
                    celeb_im_to_pass, something = original_G.synthesis(image_to_use, mask, w_celeb, return_stg1=True) 
                
                celeb_G = generated_celeb_images
                real_images_batch = image.to(global_config.device)

                

                mask2 = RandomMask(512) # adjust the masking ratio by using 'hole_range'
                mask2 = torch.from_numpy(mask2).float().to(global_config.device).unsqueeze(0)
                w_noise = torch.randn_like(w_pivot)
                noised_piv = w_pivot #+ w_noise

                generated_images, im = self.G.synthesis(real_images_batch, mask, noised_piv, return_stg1=True)
                loss1, l2_loss_val, loss_lpips = self.calc_loss(generated_images , real_images_batch , self.G)
                #loss3, l2_loss_val_3, loss_lpips3 = self.calc_loss(im, real_images_batch, self.G)
                loss2, l22, l23 = self.calc_loss(celeb_im_to_pass, celeb_G, self.G)
                self.optimizer.zero_grad()
                (loss1 + loss2).backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0
                global_config.training_step += 1
                self.image_counter += 1
                j += 1
        if self.use_wandb:
            log_images_from_w(w_pivots, self.G, [image[0] for image in images])
        snapshot_pkl = None
        snapshot_data = None
        end = time.time()
        print("runtime is: ", end - start)

        for name, module in [('G', self.G)]:
            if module is not None:
                module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
        snapshot_pkl = 'FILE-NAME'
        with open(snapshot_pkl, 'wb') as f:
            pickle.dump(module, f)
