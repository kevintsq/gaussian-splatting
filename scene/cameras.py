#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms

from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.pose_utils import interpolation_linear


class Camera:
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params,
                 image, mask, invdepthmap, image_name, uid,
                 trans=None, scale=1.0, data_device="cuda",
                 train_test_exp=False, is_test_dataset=False, is_test_view=False,
                 control_pts_num=2, interpolation_func=interpolation_linear):
        if trans is None:
            trans = np.array([0.0, 0.0, 0.0])
        w, h = resolution
        transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
        ])
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.interpolation_func = interpolation_func

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = transform(image).to(self.data_device)
        self.image_width = w
        self.image_height = h

        self.alpha_mask = torch.tensor(np.array(mask), device=self.data_device)
        self.alpha_mask = (self.alpha_mask[..., 0] > 0).float()
        if self.alpha_mask.shape != (h, w):
            self.alpha_mask = F.interpolate(self.alpha_mask[None, None], (h, w), mode='nearest-exact')[0, 0]

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image *= self.alpha_mask

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False

                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.tensor(self.invdepthmap[None], device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).T.cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar,
                                                     fovX=self.FoVx, fovY=self.FoVy).T.cuda()
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.gaussian_trans = torch.nn.Parameter(
            (torch.zeros([control_pts_num, 6], device="cuda", requires_grad=True)))

        self.pose_optimizer = torch.optim.Adam([
            {'params': [self.gaussian_trans],
             'lr': 1.e-3, "name": "translation offset"},
        ], lr=0.0, eps=1e-15)

    def update(self, global_step):
        self.pose_optimizer.step()
        # self.depth_optimizer.step()
        self.pose_optimizer.zero_grad(set_to_none=True)
        # self.depth_optimizer.zero_grad(set_to_none=True)
        decay_rate_pose = 0.01
        pose_lrate = 1e-3
        lrate_decay = 200
        decay_steps = lrate_decay * 1000
        new_lrate_pose = pose_lrate * (decay_rate_pose ** (global_step / decay_steps))
        for param_group in self.pose_optimizer.param_groups:
            param_group['lr'] = new_lrate_pose

    def get_gaussian_trans(self, alpha=0):
        return self.interpolation_func(self.gaussian_trans, alpha)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = self.world_view_transform.inverse()[3, :3]
