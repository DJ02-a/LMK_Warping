import glob
import os
import random

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from innerverz import Data_Process
from PIL import Image
from torchvision import transforms

from lib import utils
from lib.dataset import DatasetInterface

from .lmk_utils import *

DP = Data_Process()


class MyDataset(DatasetInterface):
    def __init__(self, CONFIG, dataset_path_list):
        super(MyDataset, self).__init__(CONFIG)
        self.img_size = CONFIG["BASE"]["IMG_SIZE"]
        self.set_tf()

        self.same_prob = CONFIG["BASE"]["SAME_PROB"]

        for name, list in dataset_path_list.items():
            self.__setattr__(name, list)

    def __getitem__(self, index):
        unit_clip_path = self.unit_clip_path_list[index]

        frame_names = os.listdir(os.path.join(unit_clip_path, "image"))
        # _index = random.sample(range(len(frame_names)),k=1)[0]
        from_name, to_name = random.sample(frame_names, k=2)

        # if _index == 0:
        #     from_name, to_name = random.sample(frame_names[:2], k=2)
        # elif _index == 4:
        #     from_name, to_name = random.sample(frame_names[-2:], k=2)
        # else:
        #     from_name, to_name = frame_names[_index], frame_names[_index]

        rotate, shear = random.randint(-5, 5), random.randint(-5, 5)
        scale_factor = [np.random.uniform(0.85, 1.15), np.random.uniform(0.85, 1.15)]
        translate_factor_x = [random.randint(-5, 5)] * 2
        translate_factor_y = [random.randint(-5, 5)] * 2
        scale_aug = iaa.Affine(
            scale={"x": scale_factor[0], "y": scale_factor[1]},
            rotate=(rotate, rotate),
            shear=(shear, shear),
            translate_px={"x": translate_factor_x, "y": translate_factor_y}
            # order=[1],  # use  bilinear interpolation (fast)
            # mode=["reflect"]
        ).augment_image

        from_face = np.array(
            Image.open(os.path.join(unit_clip_path, "image", from_name))
            .convert("RGB")
            .resize((self.img_size, self.img_size))
        )
        from_lmk_vis = np.array(
            Image.open(os.path.join(unit_clip_path, "lmks_756_vis", from_name))
            .convert("RGB")
            .resize((self.img_size, self.img_size))
        )
        # from_lmks = np.load(os.path.join(unit_clip_path, 'lmks_756_np', from_name.split('.')[0]+'.npy'))
        # edge_points, nearest_points, nearest_point_indexes = get_nearest_point_index(from_lmks)
        # from_lmk_vis = get_contour_vis(from_lmk_vis, edge_points, nearest_points, colors)

        from_lmk_vis = scale_aug(from_lmk_vis)
        from_lmk_vis = Image.fromarray(from_lmk_vis)
        from_lmk_vis = self.tf_color(from_lmk_vis)

        from_face = scale_aug(from_face)
        from_face = Image.fromarray(from_face)
        from_face = self.tf_color(from_face)

        to_face = DP.image_pp(
            os.path.join(unit_clip_path, "image", to_name),
            size=self.img_size,
            batch=False,
            device="cpu",
            normalize=True,
        )
        # to_lmk_vis = DP.image_pp(os.path.join(unit_clip_path, 'lmks_756_vis', to_name), size=self.img_size, batch=False, device='cpu', normalize=True)
        to_lmk_vis = np.array(
            Image.open(os.path.join(unit_clip_path, "lmks_756_vis", to_name))
            .convert("RGB")
            .resize((self.img_size, self.img_size))
        )
        # to_lmks = np.load(os.path.join(unit_clip_path, 'lmks_756_np',  to_name.split('.')[0]+'.npy'))[:,:-1]
        # to_nearest_points = to_lmks[nearest_point_indexes]
        # to_lmk_vis = get_contour_vis(to_lmk_vis, edge_points, to_nearest_points, colors)

        to_lmk_vis = Image.fromarray(to_lmk_vis)
        to_lmk_vis = self.tf_color(to_lmk_vis)

        return [from_face, to_face, from_lmk_vis, to_lmk_vis]

    def __len__(self):
        return len(self.unit_clip_path_list)

    # override
    def set_tf(self):
        self.tf_gray = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )

        self.tf_color = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )


def divide_datasets(model, CONFIG):
    unit_clip_path_list = []
    # dataset_root = '/data1/KTH-dataset/KTH_dataset_align_warp_1024'
    dataset_root = "/data1/KTH-dataset/KTH_dataset_align_warp_1024"
    # dataset_root = '/data1/KTH-dataset/KTH_dataset_unalign_warp'
    for id_num in os.listdir(dataset_root):
        for unit_clip in os.listdir(os.path.join(dataset_root, id_num)):
            unit_clip_path = os.path.join(dataset_root, id_num, unit_clip)
            if (
                len(os.listdir(unit_clip_path + "/image")) == 5
                and len(os.listdir(unit_clip_path + "/lmks_756_vis")) == 5
            ):
                unit_clip_path_list.append(unit_clip_path)

    model.train_dataset_dict = {
        "unit_clip_path_list": unit_clip_path_list[: -1 * CONFIG["BASE"]["VAL_SIZE"]],
    }
    model.valid_dataset_dict = {
        "unit_clip_path_list": unit_clip_path_list[-1 * CONFIG["BASE"]["VAL_SIZE"] :],
    }
