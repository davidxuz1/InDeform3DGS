import os
import sys
import argparse
import torch
import cv2
import shutil
import numpy as np
from pathlib import Path
import re
import subprocess
from stages.detection.detection_stage import SurgicalToolsDetectionStage
from stages.dehaze.dehaze_video import DehazeStage
from stages.segmentation.segmentation_stage import SegmentationStage
from stages.inpainting.inpainting_stage import InpaintingStage
from stages.depth.depth_stage import DepthStage
from stages.pose.pose_stage import PoseStage
from stages.gaussian.frame_resize import FrameResizer
import time
from datetime import datetime

# Get absolute project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Surgical Twin Pipeline")
    parser.add_argument("--input_video", type=str,
                      default=os.path.join(PROJECT_ROOT, "data/input/video.mp4"),
                      help="Path to input video")
    
    # Stages argument
    parser.add_argument("--stages", type=str, default="all",
                      help="Stages to run: all | segmentation | dehaze | detection | depth | inpainting | pose | gaussian")

    # Detection stage
    parser.add_argument("--model_path_detection", type=str,
                      default=os.path.join(PROJECT_ROOT, "models/pretrained/Surgical_Tools_Detection_Yolov11_Model/surgical_tools_detection_model.pt"),
                      help="Path to YOLO model weights")
    parser.add_argument("--threshold_detection", type=float,
                      default=0.6,
                      help="Detection confidence threshold (0-1)")
    parser.add_argument("--dilation_factor_detection", type=float,
                      default=1.1,
                      help="Bounding box dilation factor (>1) for detection stage")
    parser.add_argument("--fixed_bbox_watermark", type=int, nargs=4,
                      help="Fixed bounding box coordinates (x_min y_min x_max y_max) for video watermark")
    
    # Segmentation stage
    parser.add_argument("--batch_size_segmentation", type=int, default=300,
                      help="Number of frames to process in each batch for segmentation stage")
    parser.add_argument("--dilatation_factor_segmentation", type=float, default=10.0,
                      help="Factor for mask dilatation for segmentation stage")
    parser.add_argument("--mask_segmentation", type=int, default=1,
                      help="1 to save binary masks, 2 to skip mask saving")
    
    # Depth stage
    parser.add_argument("--encoder_depth", type=str, default='vitl',
                    choices=['vits', 'vitb', 'vitl', 'vitg'],
                    help="Encoder type for depth estimation")
    
    # Pose stage
    parser.add_argument("--image_size_pose", type=int, default=224, 
                    help="Image size for pose estimation [224 or 512]")
    parser.add_argument('--num_frames_pose', type=int, default=300, 
                        help='Maximum number of frames for video processing in pose stage')


    # Gaussian stage
    parser.add_argument("--lambdas", type=float, nargs=7, 
                      default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                      help="Values for the 7 lambda parameters: similarity, depth_inpaint, illumination, diversity, edge_smoothing, time_consistency, noise")
    parser.add_argument('--type_gaussian', type=int, choices=[1, 3], default=1, 
                    help='Type for Gaussian stage (1 or 3)')
    
    args = parser.parse_args()
    
    # Validate arguments

    if not os.path.exists(args.input_video):
        sys.exit(f"Error: Input video not found at {args.input_video}")
    
    valid_stages = [
        "all",  # all stages
        "segmentation",
        "dehaze",
        "detection",
        "inpainting",
        "depth",
        "pose",
        "gaussian"
    ]

    
    if args.stages not in valid_stages:
        sys.exit(f"Error: Invalid --stages value. Must be one of: {', '.join(valid_stages)}")

    # Detection stage validations
    if args.stages in ["all", "detection"]:
        if not os.path.exists(args.model_path_detection):
            sys.exit(f"Error: Model weights for detection stage not found at {args.model_path_detection}")
        if not 0 <= args.threshold_detection <= 1:
            sys.exit(f"Error: Threshold detection stage must be between 0 and 1")
        if args.dilation_factor_detection <= 1:
            sys.exit(f"Error: Dilation factor for detection stage must be greater than 1")
        if args.fixed_bbox_watermark is not None and len(args.fixed_bbox_watermark) != 4:
            sys.exit(f"Error: Fixed bounding box of video watermark must have 4 coordinates")

    # Segmentation stage validations
    if args.stages in ["all", "segmentation"]:
        if args.dilatation_factor_segmentation <= 1:
            sys.exit(f"Error: Dilation factor for segmentation stage must be greater than 1")
        if args.mask_segmentation not in [1, 2]:
            sys.exit(f"Error: Mask saving option in segmentation stage must be 1 or 2")
        if args.batch_size_segmentation <= 0:
            sys.exit(f"Error: Batch size for segmentation stage must be greater than 0")

    # Depth stage validations
    if args.stages in ["all", "depth"]:
        if args.encoder_depth not in ['vits', 'vitb', 'vitl', 'vitg']:
            sys.exit(f"Error: Encoder type must be one of ['vits', 'vitb', 'vitl', 'vitg']")


    # Pose stage validations
    if args.stages in ["all", "pose"]:
        if args.image_size_pose not in [224, 512]:
            sys.exit(f"Error: Image size for pose stage must be 224 or 512")
        if args.num_frames_pose <= 0:
            sys.exit(f"Error: Number of frames for pose stage must be greater than 0")

    # Gaussian stage validations
    if args.stages in ["all", "gaussian"]:
        if args.type_gaussian not in [1, 3]:
            sys.exit(f"Error: Type for Gaussian stage must be 1 or 3")
    
    return args


def log_time(file_path, stage_name, time_taken):
    with open(file_path, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} - {stage_name}: {time_taken:.2f} seconds\n")

def main():
    args = parse_args()
    total_start_time = time.time()

    # Create a log file for processing times
    log_dir = os.path.join(PROJECT_ROOT, "data/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "processing_times.txt")

    # Fixed paths
    detection_output_dir = os.path.join(PROJECT_ROOT, "data/intermediate/detection")
    dehaze_output_dir = os.path.join(PROJECT_ROOT, "data/intermediate/dehaze")
    segmentation_output_dir = os.path.join(PROJECT_ROOT, "data/intermediate/segmentation")
    inpainting_output_dir = os.path.join(PROJECT_ROOT, "data/intermediate/inpainting")
    depth_output_dir = os.path.join(PROJECT_ROOT, "data/intermediate/depth")
    pose_output_dir = os.path.join(PROJECT_ROOT, "data/intermediate/pose")
    gaussian_output_dir = os.path.join(PROJECT_ROOT, "data/output")
    os.makedirs(detection_output_dir, exist_ok=True)
    os.makedirs(dehaze_output_dir, exist_ok=True)
    os.makedirs(segmentation_output_dir, exist_ok=True)
    os.makedirs(inpainting_output_dir, exist_ok=True)
    os.makedirs(depth_output_dir, exist_ok=True)
    os.makedirs(pose_output_dir, exist_ok=True)
    os.makedirs(gaussian_output_dir, exist_ok=True)

    # Output path for detection stage
    detected_tools_video = os.path.join(detection_output_dir, "surgical_tools_detected.mp4")
    tools_bbox_file = os.path.join(detection_output_dir, "surgical_tools_bbox.txt")

    # Output path for dehaze stage
    dehazed_video = os.path.join(dehaze_output_dir, "dehazed_video.mp4")

    # Output path for segmentation stage
    segmented_video = os.path.join(segmentation_output_dir, "segmented_video.mp4")
    segmented_masks = os.path.join(segmentation_output_dir, "segmented_video_masks.npy")
    segmented_binary_masks_dir = os.path.join(segmentation_output_dir, "binary_masks")

    # Output path for inpainting stage
    inpainted_video = os.path.join(inpainting_output_dir, "inpainted_video.mp4")

    # Output path for depth stage
    depth_frames_dir_inpainted = os.path.join(depth_output_dir, "depth_frames_inpainted")
    depth_frames_dir_dehazed = os.path.join(depth_output_dir, "depth_frames_dehazed")

    # Output path for pose stage
    poses_bounds = os.path.join(pose_output_dir, "poses_bounds.npy")
    pose_output_dir = os.path.join(pose_output_dir, "pose_output")


    ####################################################################################
    ###########################SurgicalToolsDetectionStage##############################
    ####################################################################################
    if args.stages in ["all", "detection"]:
        print("\n" + "="*80)
        print("Starting Surgical Tools Detection Stage")
        print(f"Input video: {args.input_video}")
        print("="*80 + "\n")

        detection_start_time = time.time()

        try:
            # Initialize detection stage
            detection_stage = SurgicalToolsDetectionStage(
                model_path=args.model_path_detection,
                threshold=args.threshold_detection,
                dilation_factor=args.dilation_factor_detection,
                fixed_bbox_watermark=args.fixed_bbox_watermark
            )
            
            # Process video
            video_output, txt_output = detection_stage.process(
                args.input_video,
                detected_tools_video,
                tools_bbox_file
            )
            
            torch.cuda.empty_cache()
            print(f"Detection video saved at: {video_output}")
            print(f"Coordinates file saved at: {txt_output}")

            detection_time = time.time() - detection_start_time
            log_time(log_file, "Tools Detection", detection_time)

            print("\n" + "="*80)
            print("Detection Process completed successfully")
            print(f"Processing time: {detection_time:.2f} seconds")
            print(f"Detection video saved at: {video_output}")
            print(f"Coordinates file saved at: {txt_output}")
            print("="*80 + "\n")

        except Exception as e:
            sys.exit(f"Error during detection: {str(e)}")


    ####################################################################################
    #################################DehazeStage########################################
    ####################################################################################
    if args.stages in ["all", "dehaze"]:
        print("\n" + "="*80)
        print("Starting Dehaze stage")
        print(f"Input video: {args.input_video}")
        print("="*80 + "\n")

        dehaze_start_time = time.time()
        
        try:
            # Initialize and run dehaze stage
            dehaze_stage = DehazeStage()
            dehazed_video_path = dehaze_stage.process(args.input_video, dehazed_video)
            
            torch.cuda.empty_cache()
            dehaze_time = time.time() - dehaze_start_time
            log_time(log_file, "Dehaze", dehaze_time)
            
            print("\n" + "="*80)
            print("Dehaze process completed successfully")
            print(f"Processing time: {dehaze_time:.2f} seconds")
            print(f"Processed video saved at: {dehazed_video_path}")
            print("="*80 + "\n")
            
        except Exception as e:
            sys.exit(f"Error during dehaze processing: {str(e)}")

    ####################################################################################
    #################################SegmentationStage##################################
    ####################################################################################
    if args.stages in ["all", "segmentation"]:
        print("\n" + "="*80)
        print("Starting Segmentation stage")
        print(f"Input video: {dehazed_video}")
        print("="*80 + "\n")

        segmentation_start_time = time.time()

        try:
            # Initialize and run segmentation stage
            sam2_model_path = os.path.join(PROJECT_ROOT, "models/pretrained/SAM2_model/sam2_hiera_tiny.pt")
            if not os.path.exists(sam2_model_path):
                sys.exit(f"Error: SAM2 model not found at {sam2_model_path}")

            segmentation_stage = SegmentationStage(
                model_path=sam2_model_path,
                batch_size=args.batch_size_segmentation,
                dilatation_factor=args.dilatation_factor_segmentation,
                save_masks=(args.mask_segmentation == 1)
            )
            
            segmented_video_path = segmentation_stage.process(
                dehazed_video,
                tools_bbox_file,
                segmented_video
            )
            
            torch.cuda.empty_cache()
            segmentation_time = time.time() - segmentation_start_time
            log_time(log_file, "Segmentation", segmentation_time)
            
            print("\n" + "="*80)
            print("Segmentation process completed successfully")
            print(f"Processing time: {segmentation_time:.2f} seconds")
            print(f"Segmented video saved at: {segmented_video_path}")
            print("="*80 + "\n")
            
        except Exception as e:
            sys.exit(f"Error during segmentation processing: {str(e)}")

    ####################################################################################
    #################################InpaintingStage####################################
    ####################################################################################

    if args.stages in ["all", "inpainting"]:
        print("\n" + "="*80)
        print("Starting Inpainting stage")
        print(f"Input video: {dehazed_video} and segmentation masks: {segmented_masks}")
        print("="*80 + "\n")

        inpainting_start_time = time.time()
        
        try:
            # Initialize and run inpainting stage
            sttn_model_path = os.path.join(PROJECT_ROOT, "models/pretrained/STTN_inpainting_model/sttn.pth")
            if not os.path.exists(sttn_model_path):
                sys.exit(f"Error: STTN model not found at {sttn_model_path}")
                
            inpainting_stage = InpaintingStage(sttn_model_path)
            inpainted_video_path = inpainting_stage.process(
                dehazed_video,
                segmented_masks,
                inpainted_video
            )
            
            torch.cuda.empty_cache()
            inpainting_time = time.time() - inpainting_start_time
            log_time(log_file, "Inpainting", inpainting_time)
            
            print("\n" + "="*80)
            print("Inpainting process completed successfully")
            print(f"Processing time: {inpainting_time:.2f} seconds")
            print(f"Inpainted video saved at: {inpainted_video_path}")
            print("="*80 + "\n")
            
        except Exception as e:
            sys.exit(f"Error during inpainting processing: {str(e)}")


    ####################################################################################
    #################################DepthStage#########################################
    ####################################################################################
    if args.stages in ["all", "depth"]:
        print("\n" + "="*80)
        print("Starting Depth stage")
        print(f"Input video: {inpainted_video}")
        print("="*80 + "\n")

        # Verify Depth-Anything model existence
        depth_model_path = os.path.join(PROJECT_ROOT, "models/pretrained/depth_model", f"depth_anything_v2_{args.encoder_depth}.pth")
        if not os.path.exists(depth_model_path):
            sys.exit(f"Error: Depth-Anything model not found at {depth_model_path}")

        depth_start_time = time.time()
        
        try:
            # Initialize and run depth stage
            depth_stage = DepthStage(
                model_path=depth_model_path,
                model_type=args.encoder_depth
            )
            # Batch size for depth estimation
            # depth_stage = DepthStage(
            #     model_path=depth_model_path,
            #     model_type=args.encoder_depth,
            #     input_size=518,
            #     batch_size=8  # Ajustar según memoria GPU
            # )
            depth_frames_path_inpainted = depth_stage.process(inpainted_video, depth_frames_dir_inpainted)
            depth_fames_path_dehazed = depth_stage.process(dehazed_video, depth_frames_dir_dehazed)
            
            torch.cuda.empty_cache()
            depth_time = time.time() - depth_start_time
            log_time(log_file, "Depth Estimation", depth_time)
            
            print("\n" + "="*80)
            print("Depth process completed successfully")
            print(f"Processing time: {depth_time:.2f} seconds")
            print(f"Depth frames of inpainted video saved at: {depth_frames_path_inpainted}")
            print(f"Depth frames of dehazed video saved at: {depth_fames_path_dehazed}")
            print("="*80 + "\n")
            
        except Exception as e:
            sys.exit(f"Error during depth processing: {str(e)}")


    ####################################################################################
    #################################PoseStage##########################################
    ####################################################################################
    if args.stages in ["all", "pose"]:
        print("\n" + "="*80)
        print("Starting Pose stage")
        print(f"Input video directory: {inpainted_video}")
        print("="*80 + "\n")

        # Initialize and run segmentation stage
        pose_model_path = os.path.join(PROJECT_ROOT, "models/pretrained/pose_model/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth")
        if not os.path.exists(pose_model_path):
            sys.exit(f"Error: Pose model not found at {pose_model_path}")
            
        pose_start_time = time.time()
        
        try:
            # Initialize and run pose stage
            pose_stage = PoseStage(
            model_path=pose_model_path,
            image_size=args.image_size_pose,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            output_dir=pose_output_dir,
            use_gt_mask=False,
            fps=0,
            num_frames=args.num_frames_pose
            )
            poses_bounds_path = pose_stage.process(
                input_video= inpainted_video,
                output_path=poses_bounds,
                output_dir=pose_output_dir
            )

            torch.cuda.empty_cache()
            pose_time = time.time() - pose_start_time
            log_time(log_file, "Pose Estimation", pose_time)
            
            print("\n" + "="*80)
            print("Pose process completed successfully")
            print(f"Processing time: {pose_time:.2f} seconds")
            print(f"Poses bounds saved at: {poses_bounds_path}")
            print("="*80 + "\n")
            
        except Exception as e:
            sys.exit(f"Error during pose processing: {str(e)}")
                

    ####################################################################################
    #################################GaussianStage######################################
    ####################################################################################
    if args.stages in ["all", "gaussian"]:
        print("\n" + "="*80)
        print("Starting Gaussian stage")
        print("Collecting and processing inputs from previous stages")
        print("="*80 + "\n")

        gaussian_start_time = time.time()

        try:
            # Import FrameResizer
            from stages.gaussian.frame_resize import FrameResizer
            
            # Create Gaussian input directory
            gaussian_input_dir = os.path.join(PROJECT_ROOT, "data/intermediate/gaussian")
            if os.path.exists(gaussian_input_dir):
                shutil.rmtree(gaussian_input_dir)
            os.makedirs(gaussian_input_dir, exist_ok=True)
            
            # Define processing configurations
            processing_configs = [
                # (source, output_folder, prefix, description)
                (dehazed_video, os.path.join(gaussian_input_dir, "images"), 1, "Dehazed video frames"),
                (inpainted_video, os.path.join(gaussian_input_dir, "images_inpainted"), 2, "Inpainted video frames"),
                (depth_frames_dir_inpainted, os.path.join(gaussian_input_dir, "depth_inpainted"), 2, "Inpainted depth frames"),
                (depth_frames_dir_dehazed, os.path.join(gaussian_input_dir, "depth"), 3, "Dehazed depth frames"),
                (segmented_binary_masks_dir, os.path.join(gaussian_input_dir, "masks"), 4, "Segmentation masks")
            ]
            
            # Process each source with FrameResizer
            for source, output_folder, prefix, description in processing_configs:
                if os.path.exists(source):
                    print(f"\nProcessing {description}...")
                    print(f"Source: {source}")
                    print(f"Destination: {output_folder}")
                    print(f"Using prefix type: {prefix}")
                    
                    # Create and run the frame resizer
                    resizer = FrameResizer(
                        input_path=source,
                        output_folder=output_folder,
                        target_size=(224, 224),  # Default size, can be adjusted if needed
                        prefix=prefix
                    )
                    resizer.resize_frames()
                else:
                    print(f"Warning: Source path does not exist: {source}")
            
            # Copy additional files that don't need resizing
            additional_files = [
                (poses_bounds, os.path.join(gaussian_input_dir, "poses_bounds.npy"))
            ]
            
            for source, destination in additional_files:
                if os.path.isfile(source):
                    shutil.copy2(source, destination)
                    print(f"Copied file: {source} -> {destination}")
                else:
                    print(f"Warning: Source file does not exist: {source}")
            
            # Copy processed data to InDeform3DGS directory
            indeform_data_dir = os.path.join(PROJECT_ROOT, "third_party/InDeform3DGS/data/own_video")
            os.makedirs(indeform_data_dir, exist_ok=True)
            
            # Create and prepare base directory
            indeform_base_dir = os.path.join(PROJECT_ROOT, "third_party/InDeform3DGS/base/own_video")
            os.makedirs(indeform_base_dir, exist_ok=True)
            
            print("\nCopying processed data to InDeform3DGS...")
            print(f"Source: {gaussian_input_dir}")
            print(f"Destination 1: {indeform_data_dir}")
            print(f"Destination 2: {indeform_base_dir}")
            
            # Copy all contents from gaussian_input_dir to indeform_data_dir
            if os.path.exists(gaussian_input_dir):
                # Remove destination directory if it exists to ensure clean copy
                if os.path.exists(indeform_data_dir):
                    shutil.rmtree(indeform_data_dir)
                
                # Copy the entire directory
                shutil.copytree(gaussian_input_dir, indeform_data_dir)
                print(f"Successfully copied data to {indeform_data_dir}")
                
                # Copy specific folders to base/own_video
                # Copy images folder to base/own_video
                images_source = os.path.join(gaussian_input_dir, "images")
                images_dest = os.path.join(indeform_base_dir, "images")
                if os.path.exists(images_source):
                    if os.path.exists(images_dest):
                        shutil.rmtree(images_dest)
                    shutil.copytree(images_source, images_dest)
                    print(f"Successfully copied images to {images_dest}")
                
                # Copy masks folder to base/own_video/gt_masks
                masks_source = os.path.join(gaussian_input_dir, "masks")
                masks_dest = os.path.join(indeform_base_dir, "gt_masks")
                if os.path.exists(masks_source):
                    if os.path.exists(masks_dest):
                        shutil.rmtree(masks_dest)
                    shutil.copytree(masks_source, masks_dest)
                    print(f"Successfully copied masks to {masks_dest} (renamed to gt_masks)")
                
                # Run iterate.py script using subprocess
                type_value = args.type_gaussian
                deform3dgs_dir = os.path.join(PROJECT_ROOT, "third_party/InDeform3DGS")
                print("\n" + "="*80)
                print(f"Running InDeform3DGS training script in {deform3dgs_dir}")
                print(f"Command: python3 iterate.py --dataset own_video --type {type_value} --lambdas {' '.join(map(str, args.lambdas))}")
                print("="*80 + "\n")

                # Change to the InDeform3DGS directory and run the script
                try:
                    # Construir el comando con los lambdas
                    subprocess_cmd = ["python3", "iterate.py", "--dataset", "own_video", "--type", str(type_value), "--lambdas"]
                    # Añadir cada valor lambda como argumento separado
                    subprocess_cmd.extend([str(lam) for lam in args.lambdas])
                    
                    subprocess_result = subprocess.run(
                        subprocess_cmd,
                        cwd=deform3dgs_dir,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    print("\nInDeform3DGS training completed successfully")
                    print("Output:")
                    print(subprocess_result.stdout)
                    
                    if subprocess_result.stderr:
                        print("Warnings/Errors:")
                        print(subprocess_result.stderr)
                    
                    # Create output directory for results
                    output_dir = os.path.join(PROJECT_ROOT, "data/output")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Delete the data directory that was copied to InDeform3DGS
                    deform3dgs_data_dir = os.path.join(deform3dgs_dir, "data")
                    if os.path.exists(deform3dgs_data_dir):
                        print(f"\nRemoving temporary data directory: {deform3dgs_data_dir}")
                        shutil.rmtree(deform3dgs_data_dir)
                        print("Data directory removed successfully")
                    
                    # Delete the base directory that was copied to InDeform3DGS
                    deform3dgs_base_dir = os.path.join(deform3dgs_dir, "base")
                    if os.path.exists(deform3dgs_base_dir):
                        print(f"\nRemoving temporary base directory: {deform3dgs_base_dir}")
                        shutil.rmtree(deform3dgs_base_dir)
                        print("Base directory removed successfully")
                    
                    # Move metric_logs and output directories to data/output
                    source_dirs = {
                        "metric_logs": os.path.join(deform3dgs_dir, "metric_logs"),
                        "output": os.path.join(deform3dgs_dir, "output")
                    }
                    
                    for dir_name, source_path in source_dirs.items():
                        if os.path.exists(source_path):
                            dest_path = os.path.join(output_dir, dir_name)
                            
                            # Remove destination if it exists
                            if os.path.exists(dest_path):
                                if os.path.isdir(dest_path):
                                    shutil.rmtree(dest_path)
                                else:
                                    os.remove(dest_path)
                            
                            print(f"\nMoving {dir_name} directory:")
                            print(f"From: {source_path}")
                            print(f"To: {dest_path}")
                            
                            # Move the directory
                            shutil.move(source_path, dest_path)
                            print(f"{dir_name} moved successfully")
                        else:
                            print(f"Warning: Source directory does not exist: {source_path}")
                    
                    print(f"\nAll results moved to: {output_dir}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"Error running InDeform3DGS script: {e}")
                    print("Output:")
                    print(e.stdout)
                    print("Error:")
                    print(e.stderr)
                    raise Exception("InDeform3DGS training failed")
                    
            else:
                print(f"Warning: Source directory does not exist: {gaussian_input_dir}")
            
            gaussian_time = time.time() - gaussian_start_time
            log_time(log_file, "Gaussian Input Preparation", gaussian_time)
            
            print("\n" + "="*80)
            print("Gaussian stage completed successfully")
            print(f"Total processing time: {gaussian_time:.2f} seconds")
            print(f"Gaussian inputs processed and collected at: {gaussian_input_dir}")
            print(f"Results saved to: {output_dir}")
            print("="*80 + "\n")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            sys.exit(f"Error during Gaussian stage: {str(e)}")



if __name__ == "__main__":
    main()











