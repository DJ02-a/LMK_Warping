import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.sub_nets.dense_motion import DenseMotionNetworkReg
from core.sub_nets.LadderEncoder import LadderEncoder
from lib.nets import ConvBlock


class OcclusionAwareSPADEGenerator(nn.Module):
    """
    Generator that given source image, source ldmk image and driving ldmk image try to transform image according to movement trajectories
    according to the ldmks.
    """

    def __init__(
        self,
        block_expansion=32,
        max_features=1024,
        with_warp_im=False,
        hasid=False,
        with_gaze_htmap=False,
        with_ldmk_line=False,
        with_mouth_line=False,
        with_ht=False,
        use_IN=False,
        use_SN=True
    ):
        super(OcclusionAwareSPADEGenerator, self).__init__()
        self.with_warp_im = with_warp_im
        self.with_gaze_htmap = with_gaze_htmap
        self.with_ldmk_line = with_ldmk_line
        self.with_mouth_line = with_mouth_line
        self.with_ht = with_ht
        self.use_IN = use_IN
        self.use_SN = use_SN

        ##
        ladder_norm_type = "spectralinstance" if use_SN else "instance"
        self.ladder_network = LadderEncoder(
            need_feat=False,
            use_mask=False,
            label_nc=0,
            z_dim=512,
            norm_type=ladder_norm_type
        )
        self.dense_motion_network = DenseMotionNetworkReg(
            label_nc=0,
            ldmkimg=True,
            occlusion=True,
            block_expansion=64,
            max_features=1024,
            num_blocks=5,
            dec_lease=2,
            Lwarp=True,
            AdaINc=512
        )


        self.apply(self._init_weights)

    def _init_weights(self, m):
        gain = 0.02
        if isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.xavier_normal_(m.weight, gain=gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.xavier_normal_(m.weight, gain=gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(
            self,
            source_image,
            ldmk_line=None
        ):
        output_dict = {}

        input_t = (
            source_image
            if ldmk_line is None
            else torch.cat((source_image, ldmk_line), dim=1)
        )
        input_t_256 = F.interpolate(input_t, (256,256))
        style_feat, _ = self.ladder_network(input_t_256)
        drv_exp = style_feat
        dense_motion = self.dense_motion_network(input_t, drv_exp)

        output_dict["deformation"] = dense_motion["deformation"]

        return output_dict, dense_motion

class MyGenerator(nn.Module):
    def __init__(self):
        super(MyGenerator, self).__init__()
        self.generator = OcclusionAwareSPADEGenerator(block_expansion=64, max_features=512, with_gaze_htmap=True, with_mouth_line=True, with_ldmk_line=True, use_IN=True, hasid=True)

    def forward(self, source_image, ldmk_line):
        output_dict, dense_motion = self.generator(source_image, ldmk_line)
        output_dict["deformed"] = self.deform_input(
            source_image, dense_motion["deformation"]
        )
        return output_dict["deformed"]
    
    
    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode="bilinear")
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(
            inp.to(deformation.dtype), deformation, padding_mode="reflection"
        )