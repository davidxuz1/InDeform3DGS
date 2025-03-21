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
import imageio
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_flow as render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, FDMHiddenParams


from arguments import DatasetParams #edit own_video

from scene.flexible_deform_model import GaussianModel
from time import time
import open3d as o3d
from utils.graphics_utils import fov2focal
import cv2


#edit
from scene.cameras import Camera


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)



def render_set(model_path, name, iteration, views, gaussians, pipeline, background,\
    no_fine, render_test=False, reconstruct=False, crop_size=0):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gtdepth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
    masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "masks")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gtdepth_path, exist_ok=True)
    makedirs(masks_path, exist_ok=True)
    
    render_images = []
    render_depths = []
    gt_list = []
    gt_depths = []
    mask_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        stage = 'coarse' if no_fine else 'fine'
        rendering = render(view, gaussians, pipeline, background)
        render_depths.append(rendering["depth"].cpu())
        render_images.append(rendering["render"].cpu())
        if name in ["train", "test", "video"]:
            gt = view.original_image[0:3, :, :]
            gt_list.append(gt)
            mask = view.mask
            mask_list.append(mask)
            gt_depth = view.original_depth
            gt_depths.append(gt_depth)
    
    if render_test:
        test_times = 20
        for i in range(test_times):
            for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
                if idx == 0 and i == 0:
                    time1 = time()
                stage = 'coarse' if no_fine else 'fine'
                rendering = render(view, gaussians, pipeline, background)
        time2=time()
        print("FPS:",(len(views)-1)*test_times/(time2-time1))
    
    count = 0
    print("writing training images.")
    if len(gt_list) != 0:
        for image in tqdm(gt_list):
            torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
            count+=1
            
    count = 0
    print("writing rendering images.")
    if len(render_images) != 0:
        for image in tqdm(render_images):
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing mask images.")
    if len(mask_list) != 0:
        for image in tqdm(mask_list):
            image = image.float()
            torchvision.utils.save_image(image, os.path.join(masks_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing rendered depth images.")
    if len(render_depths) != 0:
        for image in tqdm(render_depths):
            image = np.clip(image.cpu().squeeze().numpy().astype(np.uint8), 0, 255)
            cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(count) + ".png"), image)
            count += 1
    
    count = 0
    print("writing gt depth images.")
    if len(gt_depths) != 0:
        for image in tqdm(gt_depths):
            image = image.cpu().squeeze().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(gtdepth_path, '{0:05d}'.format(count) + ".png"), image)
            count += 1
            
    # render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
    # render_array = (render_array*255).clip(0, 255).cpu().numpy().astype(np.uint8) # BxHxWxC
    # imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'ours_video.mp4'), render_array, fps=30, quality=8)
    
    # gt_array = torch.stack(gt_list, dim=0).permute(0, 2, 3, 1)
    # gt_array = (gt_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)
    # imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'gt_video.mp4'), gt_array, fps=30, quality=8)

    #edit
    # Solo crear videos si hay ground truth
    if len(gt_list) > 0:
        render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
        render_array = (render_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'ours_video.mp4'), render_array, fps=30, quality=8)
        
        gt_array = torch.stack(gt_list, dim=0).permute(0, 2, 3, 1)
        gt_array = (gt_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'gt_video.mp4'), gt_array, fps=30, quality=8)
    else:
        # Solo crear video de renders para vistas personalizadas
        render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
        render_array = (render_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'custom_views.mp4'), render_array, fps=30, quality=8)

    FoVy, FoVx, height, width = view.FoVy, view.FoVx, view.image_height, view.image_width
    focal_y, focal_x = fov2focal(FoVy, height), fov2focal(FoVx, width)
    camera_parameters = (focal_x, focal_y, width, height)
    

    if reconstruct:
        print('file name:', name)
        reconstruct_point_cloud(render_images, mask_list, render_depths, camera_parameters, name, crop_size)
    
# def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, reconstruct_train: bool, reconstruct_test: bool, reconstruct_video: bool):
#     with torch.no_grad():
#         gaussians = GaussianModel(dataset.sh_degree, hyperparam)
#         scene = Scene(dataset, gaussians, load_iteration=iteration)

#         bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
#         background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
#         if not skip_train:
#             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, False, reconstruct=reconstruct_train)
#         if not skip_test:
#             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, False, reconstruct=reconstruct_test, crop_size=20)
#         if not skip_video:
#             render_set(dataset.model_path,"video",scene.loaded_iter, scene.getVideoCameras(),gaussians,pipeline,background, False, render_test=True, reconstruct=reconstruct_video, crop_size=20)



# #edit
# def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, reconstruct_train: bool, reconstruct_test: bool, reconstruct_video: bool):
#     with torch.no_grad():
#         gaussians = GaussianModel(dataset.sh_degree, hyperparam)
#         scene = Scene(dataset, gaussians, load_iteration=iteration)
        
#         bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
#         background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
#         # # Obtener todas las cámaras base para obtener los tiempos
#         # base_cameras = scene.getTestCameras()
#         # for i, cam in enumerate(base_cameras):
#         #     position = -cam.R.T @ cam.T
#         #     forward = -cam.R[:, 2]
#         #     look_at = position + forward
#         #     up = cam.R[:, 1]
#         #     print(f"Camera {i}:")
#         #     print(f"Position: {position}")
#         #     print(f"Look at: {look_at}")
#         #     print(f"Up vector: {up}")

#         # # Crear una única cámara fija con los mismos parámetros que las originales
#         # fixed_position = np.array([-0.8, 0.2, 0.6])
#         # fixed_look_at = np.array([-0.7, 0.1, 1.0])  # Mismo look_at que las cámaras originales
#         # fixed_up = np.array([0.0, -1.0, 0.0])      # Mismo up vector que las cámaras originales
#         # fixed_fovx = 1.0239093368021417
#         # fixed_fovy = 0.8449463442193673
#         # fixed_fovx = 2
#         # fixed_fovy = 2
        
#         # # Crear la secuencia temporal completa desde la cámara fija
#         # custom_views = []
#         # print(f"Creando secuencia desde posición fija: {fixed_position}")
        
#         # # Usar los tiempos de las cámaras originales
#         # for base_cam in base_cameras:
#         #     custom_cam = create_custom_camera(
#         #         position=fixed_position,
#         #         look_at=fixed_look_at,
#         #         up=fixed_up,
#         #         fovx=fixed_fovx,
#         #         fovy=fixed_fovy,
#         #         width=base_cam.image_width,
#         #         height=base_cam.image_height,
#         #         time=base_cam.time
#         #     )
#         #     custom_views.append(custom_cam)
        
#         # # Renderizar la secuencia completa desde la cámara fija
#         # print(f"Renderizando secuencia desde cámara fija...")
#         # render_set(dataset.model_path, "fixed_view", scene.loaded_iter, 
#         #           custom_views, gaussians, pipeline, background, False)
        
#         # Renderizar vistas originales si no se han saltado
#         # if not skip_train:
#         #     render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, False, reconstruct=reconstruct_train)
#         # if not skip_test:
#         #     render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, False, reconstruct=reconstruct_test, crop_size=20)
#         if not skip_video:
#             render_set(dataset.model_path,"video",scene.loaded_iter, scene.getVideoCameras(),gaussians,pipeline,background, False, render_test=True, reconstruct=reconstruct_video, crop_size=20)


# edit own_video
def render_sets(dataset: ModelParams, hyperparam, iteration: int, pipeline: PipelineParams, dataset_params, skip_train: bool, skip_test: bool, skip_video: bool, reconstruct_train: bool, reconstruct_test: bool, reconstruct_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        # Añadir dataset_params a dataset en lugar de pasarlo a Scene
        dataset.DatasetParams = dataset_params
        scene = Scene(dataset, gaussians, load_iteration=iteration)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_video:
            render_set(dataset.model_path, "video", scene.loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background, False, render_test=True, reconstruct=reconstruct_video, crop_size=20)

def create_custom_camera(position, look_at, up, fovx, fovy, width, height, time):
    # Calcular R y T desde position, look_at y up
    forward = look_at - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    R = np.stack([right, up, -forward], axis=1)
    T = -R @ position

    # Crear una cámara usando la clase Camera existente
    return Camera(
        colmap_id=-1,
        R=R,
        T=T,
        FoVx=fovx,
        FoVy=fovy,
        image=torch.zeros((3, height, width)),
        depth=torch.zeros((1, height, width)),
        mask=torch.ones((1, height, width)),
        gt_alpha_mask=None,
        image_name="custom_view",
        uid=-1,
        time=time
    )





def reconstruct_point_cloud(images, masks, depths, camera_parameters, name, crop_left_size=0):
    import cv2
    import copy
    output_frame_folder = os.path.join("reconstruct", name)
    os.makedirs(output_frame_folder, exist_ok=True)
    frames = np.arange(len(images))
    # frames = [0]
    focal_x, focal_y, width, height = camera_parameters
    if crop_left_size > 0:
        width = width - crop_left_size
        height = height - crop_left_size//2
    for i_frame in frames:
        rgb_tensor = images[i_frame]
        rgb_np = rgb_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to("cpu").numpy()
        depth_np = depths[i_frame].cpu().numpy()
        if len(depth_np.shape) == 3:
            depth_np = depth_np[0]
        # depth_np = depth_np.squeeze(0)
        if crop_left_size > 0:
            rgb_np = rgb_np[:, crop_left_size:, :]
            depth_np = depth_np[:, crop_left_size:]
            rgb_np = rgb_np[:-crop_left_size//2, :, :]
            depth_np = depth_np[:-crop_left_size//2, :]
            
        # mask = masks[i_frame]
        # mask = mask.squeeze(0).cpu().numpy()
        
        rgb_new = copy.deepcopy(rgb_np)
        # depth_np[mask == 0] =0
        # rgb_new[mask ==0] = np.asarray([0,0,0]) 
        depth_smoother = (32, 64, 32) # (128, 64, 64) #[24, 64, 32]
        # print(depth_np.shape)
        depth_np = cv2.bilateralFilter(depth_np, depth_smoother[0], depth_smoother[1], depth_smoother[2])
        
        close_depth = np.percentile(depth_np[depth_np!=0], 5)
        inf_depth = np.percentile(depth_np, 95)
        depth_np = np.clip(depth_np, close_depth, inf_depth)

        rgb_im = o3d.geometry.Image(rgb_new.astype(np.uint8))
        depth_im = o3d.geometry.Image(depth_np)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(int(width), int(height), focal_x, focal_y, width / 2, height / 2),
            project_valid_depth_only=True
        )
        o3d.io.write_point_cloud(os.path.join(output_frame_folder, 'frame_{}.ply'.format(i_frame)), pcd)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = FDMHiddenParams(parser)
    dp = DatasetParams(parser) #edit own_video
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--reconstruct_train", action="store_true")
    parser.add_argument("--reconstruct_test", action="store_true")
    parser.add_argument("--reconstruct_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering ", args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    #edit own_video
    render_sets(model.extract(args), 
           hyperparam.extract(args), 
           args.iteration, 
           pipeline.extract(args),
           dp.extract(args), 
           args.skip_train, args.skip_test, args.skip_video,
           args.reconstruct_train,args.reconstruct_test,args.reconstruct_video)
     
    # render_sets(model.extract(args), hyperparam.extract(args), args.iteration, 
    #     pipeline.extract(args), 
    #     args.skip_train, args.skip_test, args.skip_video,
    #     args.reconstruct_train,args.reconstruct_test,args.reconstruct_video)