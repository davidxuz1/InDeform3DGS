#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import random
import os 
import torch
from random import randint
from utils.loss_utils import l1_loss
from gaussian_renderer import render_flow as render

import sys
from scene import  Scene
from scene.flexible_deform_model import GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
#from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments import ModelParams, PipelineParams, OptimizationParams, DatasetParams # edit own_data
from arguments import FDMHiddenParams as ModelHiddenParams
from utils.timer import Timer
import torch.nn.functional as F

# import lpips
from utils.scene_utils import render_training_image

# edit
import torchvision.models as models
import torch.nn as nn




to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# edit
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features[:23]  # Usar hasta conv4_3
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False
        self.features = self.features.cuda()

    def forward(self, x):
        return self.features(x)

# Inicializar el extractor (esto va al principio del código)
vgg_features = VGGFeatureExtractor()
epsilon = 1e-6  # Para evitar división por cero

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, train_iter, timer):
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None

    
    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    
    # lpips_model = lpips.LPIPS(net="vgg").cuda()
    video_cams = scene.getVideoCameras()
    
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()
        
    for iteration in range(first_iter, final_iter+1):        

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cams = [viewpoint_stack[idx]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []
        
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, depth, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda().float()
            gt_depth = viewpoint_cam.original_depth.cuda().float()
            mask = viewpoint_cam.mask.cuda()
            
            images.append(image.unsqueeze(0))
            depths.append(depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        depth_tensor = torch.cat(depths, 0)
        gt_image_tensor = torch.cat(gt_images,0)
        gt_depth_tensor = torch.cat(gt_depths, 0)
        mask_tensor = torch.cat(masks, 0)
        
        Ll1 = l1_loss(image_tensor, gt_image_tensor, mask_tensor)
        
        if (gt_depth_tensor!=0).sum() < 10:
            depth_loss = torch.tensor(0.).cuda()
        else:
            depth_tensor[depth_tensor!=0] = 1 / depth_tensor[depth_tensor!=0]
            gt_depth_tensor[gt_depth_tensor!=0] = 1 / gt_depth_tensor[gt_depth_tensor!=0]
     
            depth_loss = l1_loss(depth_tensor, gt_depth_tensor, mask_tensor)
    
      
        psnr_ = psnr(image_tensor, gt_image_tensor, mask_tensor).mean().double()        
        loss = Ll1 + depth_loss 
        




        if viewpoint_cam.has_inpainted:
            #print("Using inpainted image")
            inpainted_data = viewpoint_cam.load_inpainted_image()
            if inpainted_data[0] is not None:
                #print("Inpainted image loaded")
                # Convertir numpy array a tensor de PyTorch y luego mover a CUDA
                inpainted_image = torch.from_numpy(inpainted_data[0]).cuda()
                if inpainted_image.shape[-1] == 3:
                    inpainted_image = inpainted_image.permute(2, 0, 1)
                inpainted_image = inpainted_image / 255.0
                inpainted_image = inpainted_image.unsqueeze(0)
                mask_expanded = mask.expand_as(image).float()  # Añadido .float()

                # # import matplotlib.pyplot as plt

                # plt.figure(figsize=(10, 8))
                # plt.imshow(mask_expanded[0].cpu().numpy(), cmap='gray')
                # plt.colorbar()
                # plt.title('Error Calculation Mask')
                # plt.savefig(f'debug_masks/error_mask_{iteration}.png')
                # plt.close()
                
                # # 1. Región de herramientas
                # tool_region = image * (1 - mask_expanded)
                # tool_region = torch.clamp(tool_region, 0.0, 1.0)
                # tool_region = tool_region.unsqueeze(0)

                tool_region = image * (1-mask_expanded)
                tool_region = torch.clamp(tool_region, 0.0, 1.0)
                tool_region = tool_region.unsqueeze(0)


                # # For tool region visualization
                # tool_region_vis = (image * (1-mask_expanded)).cpu().detach().numpy()
                # plt.figure(figsize=(10, 8))
                # # Check shape before transposing
                # if len(tool_region_vis.shape) == 3:
                #     plt.imshow(np.transpose(tool_region_vis, (1, 2, 0)))
                # else:
                #     plt.imshow(tool_region_vis, cmap='gray')
                # plt.title('Tool Region')
                # plt.savefig(f'debug_masks/tool_region_{iteration}.png')
                # plt.close()

                # # 2. Pérdida de similaridad con imagen inpainted
                if opt.lambda_similarity > 0.0:
                    similarity_loss = l1_loss(tool_region, inpainted_image * (1 - mask_expanded))
                    loss += opt.lambda_similarity * similarity_loss
                    # print(f"[ITER {iteration}] Similarity loss: {similarity_loss.item():.4f}")
                    torch.cuda.empty_cache()

                # # 3. Pérdida de profundidad en región inpainted
                if opt.lambda_depth_inpaint > 0.0:
                    if inpainted_data[1] is not None and depth is not None:
                        # Convertir numpy array a tensor de PyTorch y luego mover a CUDA
                        inpainted_depth = torch.from_numpy(inpainted_data[1]).cuda()

                        depth_inpaint_loss = l1_loss(
                            depth * (1 - mask_expanded[0:1]),
                            inpainted_depth * (1 - mask_expanded[0:1])
                        )
                        # print(f"[ITER {iteration}] Depth inpaint loss: {depth_inpaint_loss.item():.4f}")
                        loss += opt.lambda_depth_inpaint * depth_inpaint_loss
                        torch.cuda.empty_cache()


                # 4. Pérdida de iluminancia usando convolución
                if opt.lambda_illumination > 0.0:
                    luminance_kernel = torch.tensor([0.299, 0.587, 0.114], device="cuda").view(1, 3, 1, 1)
                    rendered_luminance = F.conv2d(tool_region, luminance_kernel)
                    inpainted_luminance = F.conv2d(inpainted_image, luminance_kernel)
                    illumination_loss = l1_loss(rendered_luminance * (1 - mask_expanded[0:1]),
                                                inpainted_luminance * (1 - mask_expanded[0:1]))
                    loss += opt.lambda_illumination * illumination_loss
                    #print(f"[ITER {iteration}] Illumination loss: {illumination_loss.item():.4f}")
                    torch.cuda.empty_cache()

                # 5. Pérdida de diversidad para evitar soluciones triviales
                if opt.lambda_diversity > 0.0:
                    tool_region_normalized = tool_region.clone()
                    tool_region_normalized = tool_region_normalized * 2 - 1
                    
                    with torch.no_grad():
                        Fi = vgg_features(tool_region_normalized)
                    
                    # Usar (1 - mask_expanded) para la región de herramientas
                    mask_resized = torch.nn.functional.interpolate(
                        (1 - mask_expanded).unsqueeze(0), 
                        size=(Fi.shape[2], Fi.shape[3]), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    mask_resized = mask_resized[:, 0:1, :, :].repeat(1, Fi.shape[1], 1, 1)
                    
                    diversity_loss = 1 / (torch.var(Fi * mask_resized) + epsilon)
                    # print(f"[ITER {iteration}] Diversity loss: {diversity_loss.item():.4f}")
                    loss += opt.lambda_diversity * diversity_loss


                # 6. Pérdida de suavizado de bordes
                if opt.lambda_edge_smoothing > 0.0:
                    kernel_size = 5
                    sigma = kernel_size / 3.0
                    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device="cuda").float()
                    gauss = torch.exp(-0.5 * (x / sigma) ** 2)
                    gauss = gauss / gauss.sum()
                    
                    # Usar (1 - mask_expanded) para la región de herramientas
                    mask_single_channel = (1 - mask_expanded)[:1]
                    
                    smoothed_mask = F.conv2d(mask_single_channel.float().unsqueeze(0), 
                                            gauss.view(1, 1, -1, 1), 
                                            padding=(kernel_size // 2, 0))
                    
                    smoothed_mask = F.conv2d(smoothed_mask, 
                                            gauss.view(1, 1, 1, -1), 
                                            padding=(0, kernel_size // 2))
                    
                    smoothed_mask = smoothed_mask.expand_as(mask_expanded.unsqueeze(0))
                    
                    edge_region = torch.abs(smoothed_mask - (1 - mask_expanded).float().unsqueeze(0))
                    
                    edge_loss = l1_loss(
                        tool_region * edge_region,
                        inpainted_image * edge_region)

                    loss += opt.lambda_edge_smoothing * edge_loss
                    #print(f"[ITER {iteration}] Edge loss: {edge_loss.item():.4f}")
                    torch.cuda.empty_cache()
                    
                    
                # 7. Time consistency loss
                if opt.lambda_time_consistency > 0.0:
                    # Get current and previous frame features
                    current_region = tool_region
                    prev_region = viewpoint_cam.previous_frame_features if hasattr(viewpoint_cam, 'previous_frame_features') else None
                    
                    #print(f"[DEBUG] Current region shape: {current_region.shape}")
                    #print(f"[DEBUG] Previous region exists: {prev_region is not None}")
                    
                    if prev_region is not None:
                        # Calculate temporal consistency loss
                        temporal_loss = l1_loss(
                            current_region * (1 - mask_expanded),
                            prev_region * (1 - mask_expanded)
                        )
                        loss += opt.lambda_time_consistency * temporal_loss
                        #print(f"[ITER {iteration}] Temporal loss: {temporal_loss.item():.4f}")
                        
                    # Actualizar las características para el siguiente frame
                    viewpoint_cam.previous_frame_features = current_region.detach().clone()
                    #print(f"[DEBUG] Updated features for next iteration")
                    
                    torch.cuda.empty_cache()
                    
                # # Add noise-based loss
                if opt.lambda_noise > 0.0:
                    # Calculate local variance to detect noise
                    kernel_size = 3
                    padding = kernel_size // 2
                    
                    # Convert tool_region to single channel if needed
                    if tool_region.shape[1] != 1:
                        tool_region_gray = tool_region.mean(dim=1, keepdim=True)
                    else:
                        tool_region_gray = tool_region
                    
                    # Calculate local mean
                    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size, device="cuda") / (kernel_size * kernel_size)
                    local_mean = F.conv2d(tool_region_gray, mean_kernel, padding=padding)
                    
                    # Calculate local variance
                    local_var = F.conv2d(tool_region_gray**2, mean_kernel, padding=padding) - local_mean**2
                    
                    # Apply mask to focus on tool region
                    noise_loss = torch.mean(local_var * mask_expanded)
                    loss += opt.lambda_noise * noise_loss
                    torch.cuda.empty_cache()




        loss.backward()
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background])
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, 'fine')
            timer.start()
            
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

  
                opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


# edit own_data
def training(dataset, hyper, opt, pipe, dataset_params, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, extra_mark):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    
    # Ahora dataset incluye los dataset_params
    dataset.DatasetParams = dataset_params
    scene = Scene(dataset, gaussians)
    
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, opt.iterations, timer)

    
# def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, extra_mark):
#     tb_writer = prepare_output_and_logger(expname)
#     gaussians = GaussianModel(dataset.sh_degree, hyper)
#     dataset.model_path = args.model_path
#     timer = Timer()
#     scene = Scene(dataset, gaussians)
#     timer.start()
#     scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
#                          checkpoint_iterations, checkpoint, debug_from,
#                          gaussians, scene, tb_writer, opt.iterations,timer)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    
    if tb_writer:
        tb_writer.add_scalar(f'train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'iter_time', elapsed, iteration)
    


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    dp = DatasetParams(parser) #edit own_data
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*500 for i in range(0,120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000,])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "endonerf/pulling_fdm")
    parser.add_argument("--configs", type=str, default = "arguments/endonerf/default.py")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # edit own_data
    training(lp.extract(args),    # dataset
         hp.extract(args),    # hyper
         op.extract(args),    # opt
         pp.extract(args),    # pipe
         dp.extract(args),    # dataset_params (nuevo)
         args.test_iterations,
         args.save_iterations,
         args.checkpoint_iterations,
         args.start_checkpoint,
         args.debug_from,
         args.expname,
         args.extra_mark)

    # training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
    #     args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args.extra_mark)
    
    # All done
    print("\nTraining complete.")
