import os
import torch
import copy
import PIL.Image
import pickle
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w


class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self):

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True

        for fname, image in tqdm(self.data_loader):
            image_name = fname[0]

            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = None

            if hyperparameters.use_last_w_pivots:
                w_pivot = self.load_inversions(w_path_dir, image_name)

            elif not hyperparameters.use_last_w_pivots or w_pivot is None:
                w_pivot = self.calc_inversions(image, image_name)

            # w_pivot = w_pivot.detach().clone().to(global_config.device)
            w_pivot = w_pivot.to(global_config.device)

            torch.save(w_pivot, f'{embedding_dir}/0.pt')
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)

            for i in tqdm(range(hyperparameters.max_pti_steps)):
                print("step", i)
                print("real images batch shape", w_pivot.shape)
                generated_images = self.forward(real_images_batch, w_pivot)
                outputf = (generated_images.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)


                PIL.Image.fromarray(outputf[0].cpu().numpy(), 'RGB').save(f'samps/im{i:02d}.png')
                #print("real images batch shape", real_images_batch.shape)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()

                '''if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break'''

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    log_images_from_w([w_pivot], self.G, [image_name])

                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1
            
            snapshot_pkl = None
            snapshot_data = None
            
            for name, module in [('G', self.G)]:
                #print(name, module)
                if module is not None:
                    #if num_gpus > 1:
                    #misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
            snapshot_pkl = 'myface.pkl'
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(module, f)

            torch.save(self.G,
                       f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pt')
