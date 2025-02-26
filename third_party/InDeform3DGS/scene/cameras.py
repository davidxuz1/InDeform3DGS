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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrix2
import os
import imageio
from arguments import DatasetParams  # Para type hints si los usas
    
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, depth, mask, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 data_device = "cuda", time = 0, Znear=None, Zfar=None, 
                 K=None, h=None, w=None, 
                 has_inpainted=False,  # edit
                 dataset_params = None # own_video
                 ):
        super(Camera, self).__init__()
        self.dataset_params = dataset_params # own_video


        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        self.mask = mask

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        self.original_image = image.clamp(0.0, 1.0)
        self.original_depth = depth
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))
        
        if Zfar is not None and Znear is not None:
            self.zfar = Zfar
            self.znear = Znear
        else:
            # ENDONERF
            self.zfar = 120.0
            self.znear = 0.01
            
            # StereoMIS
            self.zfar = 250
            self.znear= 0.03
            
        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        if K is None or h is None or w is None:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        else:
            self.projection_matrix = getProjectionMatrix2(znear=self.znear, zfar=self.zfar, K=K, h = h, w=w).transpose(0,1)
        
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


        #edit
        self.has_inpainted = has_inpainted
        self.inpainted_image = None
        self.inpainted_depth = None

        #edit
        self.previous_frame_features = None
    
    #edit
    def update_temporal_features(self, features):
        """Actualiza las características del frame anterior"""
        self.previous_frame_features = features.detach().clone()
    #edit
    def get_temporal_features(self):
        """Obtiene las características del frame anterior"""
        return self.previous_frame_features
    
    
            
    #edit
    def load_inpainted_image(self):
        try:
            # # Usar la ruta base completa
            # base_path = "data/endonerf_full_datasets/pulling_soft_tissues"  # Ruta base completa EndoNeRF
            # base_path = "data/own_data"  # Ruta base completa #own_video      
            # Usar dataset_dir desde DatasetParams
            base_path = self.dataset_params.dataset_dir
            #print(f'El path base es: {base_path}')
            img_name = os.path.basename(self.image_name)
            
            # Construir rutas completas
            inpainted_img_path = os.path.join(base_path, "images_inpainted", f"{img_name}.png")
            inpainted_depth_path = os.path.join(base_path, "depth_inpainted", f"{img_name}.png")

            
            # print(f"Base path: {base_path}")
            # print(f"Image name: {img_name}")
            # print(f"Full inpainted path: {inpainted_img_path}")
            
            inpainted_image = inpainted_depth = None
            
            if os.path.exists(inpainted_img_path):
                inpainted_image = imageio.imread(inpainted_img_path).astype(np.float32)
            else:
                print(f"Inpainted image path does not exist: {inpainted_img_path}")
                
            if os.path.exists(inpainted_depth_path):
                inpainted_depth = imageio.imread(inpainted_depth_path).astype(np.float32)
            else:
                print(f"Inpainted depth path does not exist: {inpainted_depth_path}")
                
            return inpainted_image, inpainted_depth
            
        except Exception as e:
            print(f"Error in load_inpainted_image: {str(e)}")
            return None, None



class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.time = time



