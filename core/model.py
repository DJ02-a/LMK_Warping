import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from innerverz import Data_Process, FaceParser
from tqdm import tqdm

from core.loss import MyModelLoss
from core.nets import MyGenerator
from core.sub_nets.util import load_ckpt
from lib import utils
from lib.discriminators import ProjectedDiscriminator
from lib.model import ModelInterface

DP = Data_Process()

def get_grad_mask(size=512):
    x_axis = np.linspace(-1, 1, size)[:, None]
    y_axis = np.linspace(-1, 1, size)[None, :]

    arr1 = np.sqrt(x_axis ** 4 + y_axis ** 4)

    x_axis = np.linspace(-1, 1, size)[:, None]
    y_axis = np.linspace(-1, 1, size)[None, :]

    arr2 = np.sqrt(x_axis ** 2 + y_axis ** 2)

    grad_mask = np.clip(1-(arr1/2+arr2/2), 0, 1) # defayk
    grad_mask = (grad_mask[:, :, None].repeat(3, axis=2) * 3).clip(0,1)
    return grad_mask

def get_gaussian_kernel(kernel_size=9, sigma=2.0, channels=1):
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean)**2., dim=-1) /\
                        (2*variance)
                    )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    return gaussian_kernel

class MyModel(ModelInterface):
    def declare_networks(self):
        self.G = MyGenerator().cuda()
        self.D = ProjectedDiscriminator().cuda()


        self.FP = FaceParser()        
        self.set_networks_train_mode()
        
        self.grad_mask = get_grad_mask(self.CONFIG['BASE']['IMG_SIZE'])
        self.grad_mask = torch.tensor(self.grad_mask, device='cuda').permute([2,0,1]).unsqueeze(0).repeat(self.CONFIG['BASE']['BATCH_PER_GPU'],1,1,1).float()
        
        self.gaussian_window = get_gaussian_kernel(kernel_size=128*2+1, channels=1).to('cuda')
        self.window = torch.ones((1,1,9,9), device='cuda', dtype=torch.float)
        # self.window = torch.ones((self.CONFIG['BASE']['BATCH_PER_GPU'],1,3,3), device=self.FP.device, dtype=torch.float)


        # PACKAGES
        load_ckpt('./ckpt/ckpt_base.pth.tar', {'generator': self.G.generator}, device='cuda', strict=False)


    def set_networks_train_mode(self):
        self.G.train()
        self.D.train()
        self.D.feature_network.eval()
        self.D.feature_network.requires_grad_(False)
        
    def set_networks_eval_mode(self):
        self.G.eval()
        self.D.eval()

    def go_step(self):
        from_face, to_face, from_lmk_vis, to_lmk_vis = self.load_next_batch(self.train_dataloader, self.train_iterator, 'train')
        
        self.train_dict["from_face"] = from_face
        self.train_dict["from_lmk_vis"] = from_lmk_vis
        self.train_dict["to_face"] = to_face
        self.train_dict["to_lmk_vis"] = to_lmk_vis

        # run G
        self.run_G(self.train_dict)

        # update G
        loss_G = self.loss_collector.get_loss_G(self.train_dict)
        self.update_net(self.opt_G, loss_G)

        # run D
        self.run_D(self.train_dict)

        # update D
        loss_D = self.loss_collector.get_loss_D(self.train_dict)
        self.update_net(self.opt_D, loss_D)
        
        to_lmk_vis_mask = torch.where(self.train_dict["to_lmk_vis"]> -1,1,0)
        
        # print images
        self.train_images = [
            self.train_dict["from_face"],
            self.train_dict["from_lmk_vis"],
            self.train_dict["to_face"],
            self.train_dict["to_lmk_vis"],
            self.train_dict["fake_to_face"],
            self.train_dict["bg_mask"],
            self.train_dict["to_mask"] * self.train_dict["fake_to_face"],
            # ((self.train_dict["to_face"] * self.train_dict['to_mask'])*(1-to_lmk_vis_mask) + self.train_dict["to_lmk_vis"]*to_lmk_vis_mask),
            # ((self.train_dict["fake_to_face"] * self.train_dict['to_mask'])*(1-to_lmk_vis_mask) + self.train_dict["to_lmk_vis"]*to_lmk_vis_mask),
            ]
        


    def run_G(self, run_dict):
        # with torch.no_grad():
        
        from_label = self.FP.get_label(F.interpolate(run_dict['from_face'],(512,512)), 512)['label']
        from_label = F.interpolate(from_label.unsqueeze(1).float(), (self.CONFIG['BASE']['IMG_SIZE'], self.CONFIG['BASE']['IMG_SIZE']))
        from_innerface_mask = torch.where(from_label <= 14, 1, 0) - torch.where(from_label==0, 1, 0)
        # from_neck_mask = torch.where(from_label == 14, 1, 0).unsqueeze(1)
        from_cloth_mask = torch.where(from_label == 16, 1, 0)
        run_dict['from_mask'] = from_innerface_mask.type(torch.float32)
        
        # run_dict['from_mask'] = F.conv2d(from_innerface_mask.unsqueeze(1).type(torch.float32), self.window, stride=1, padding=4).clip(0,1)
        # run_dict['from_mask'] = self.grad_mask
        
        
        to_label = self.FP.get_label(F.interpolate(run_dict['to_face'],(512,512)), 512)['label']
        to_label = F.interpolate(to_label.unsqueeze(1).float(), (self.CONFIG['BASE']['IMG_SIZE'], self.CONFIG['BASE']['IMG_SIZE']))
        to_innerface_mask = torch.where(to_label <= 14, 1, 0) + torch.where(to_label == 17, 1, 0) - torch.where(to_label==0, 1, 0)
        # to_neck_mask = torch.where(to_label == 14, 1, 0).unsqueeze(1)
        # to_cloth_mask = torch.where(to_label == 16, 1, 0).unsqueeze(1)
        tmp = []
        for mask in from_cloth_mask:
            mask_np = mask.clone().detach().squeeze().cpu().numpy()
            mask_blur = DP.mask_pp(mask_np, dilate_iter=5, blur_ksize=32)
            mask_ts = torch.tensor(mask_blur, device=to_label.device).unsqueeze(0)
            tmp.append(mask_ts)
        blur_from_cloth_mask = torch.cat(tmp, dim=0).unsqueeze(1)
        
        tmp = []
        for mask in to_innerface_mask:
            mask_np = mask.clone().detach().squeeze().cpu().numpy()
            mask_blur = DP.mask_pp(mask_np, dilate_iter=5, blur_ksize=32)
            mask_ts = torch.tensor(mask_blur, device=to_label.device).unsqueeze(0)
            tmp.append(mask_ts)
        blur_to_innerface_mask = torch.cat(tmp, dim=0).unsqueeze(1)
        
        
        run_dict['to_mask'] = blur_to_innerface_mask.type(torch.float32)
        # run_dict['to_mask'] = F.conv2d(to_innerface_mask.unsqueeze(1).type(torch.float32), self.window, stride=1, padding=4).clip(0,1)
        # run_dict['to_mask'] = self.grad_mask
        
        run_dict['bg_mask'] = (1 - (self.grad_mask - blur_from_cloth_mask).clip(0,1))
        
        cat_lmk = torch.cat((run_dict['from_lmk_vis'], run_dict['to_lmk_vis']), dim=1)
        
        
        run_dict['fake_to_face'] = self.G(run_dict['from_face'], cat_lmk)
        g_pred_fake, feat_fake = self.D(run_dict["fake_to_face"] * run_dict['to_mask'], None)
        feat_real = self.D.get_feature(run_dict["to_face"] * run_dict['to_mask'])


        run_dict['g_feat_fake'] = feat_fake
        run_dict['g_feat_real'] = feat_real
        run_dict["g_pred_fake"] = g_pred_fake

    def run_D(self, run_dict):
        d_pred_real, _  = self.D(run_dict['to_face'] * run_dict['to_mask'].detach(), None)
        d_pred_fake, _  = self.D(run_dict['fake_to_face'].detach()  * run_dict['to_mask'].detach(), None)
        
        run_dict["d_pred_real"] = d_pred_real
        run_dict["d_pred_fake"] = d_pred_fake

    def do_validation(self):
        self.valid_images = []
        self.set_networks_eval_mode()

        self.loss_collector.loss_dict["valid_L_G"],  self.loss_collector.loss_dict["valid_L_D"] = 0., 0.
        pbar = tqdm(range(len(self.valid_dataloader)), desc='Run validate..')
        for _ in pbar:
            from_face, to_face, from_lmk_vis, to_lmk_vis = self.load_next_batch(self.valid_dataloader, self.valid_iterator, 'valid')
            
            self.valid_dict["from_face"] = from_face
            self.valid_dict["from_lmk_vis"] = from_lmk_vis
            self.valid_dict["to_face"] = to_face
            self.valid_dict["to_lmk_vis"] = to_lmk_vis

            with torch.no_grad():
                self.run_G(self.valid_dict)
                self.run_D(self.valid_dict)
                self.loss_collector.get_loss_G(self.valid_dict, valid=True)
                self.loss_collector.get_loss_D(self.valid_dict, valid=True)
                            
            if len(self.valid_images) < 8 : utils.stack_image_grid([
                self.valid_dict["from_face"],
                self.valid_dict["from_lmk_vis"],
                self.valid_dict["to_face"],
                self.valid_dict["to_lmk_vis"],
                self.valid_dict["fake_to_face"],
                (abs(self.valid_dict["fake_to_face"] - self.valid_dict["to_face"])),
                ],
                self.valid_images)
            
        self.loss_collector.loss_dict["valid_L_G"] /= len(self.valid_dataloader)
        self.loss_collector.loss_dict["valid_L_D"] /= len(self.valid_dataloader)
        self.loss_collector.val_print_loss()
        
        self.valid_images = torch.cat(self.valid_images, dim=-1)

        self.set_networks_train_mode()
        
    def do_test(self):
        self.test_images = []
        self.set_networks_eval_mode()
        
        pbar = tqdm(range(len(self.test_dataloader)), desc='Run test...')
        for _ in pbar:
            source, GT = self.load_next_batch(self.test_dataloader, self.test_iterator, 'test')
            
            self.test_dict["source"] = source
            self.test_dict["GT"] = GT

            with torch.no_grad():
                self.run_G(self.test_dict)
                self.run_D(self.test_dict)

            utils.stack_image_grid([self.test_dict["source"], self.test_dict["output"], self.test_dict["GT"]], self.test_images)
        
        self.test_images = torch.cat(self.test_images, dim=-1)

        self.set_networks_train_mode()

    @property
    def loss_collector(self):
        return self._loss_collector


    def set_loss_collector(self):
        self._loss_collector = MyModelLoss(self.CONFIG)        
