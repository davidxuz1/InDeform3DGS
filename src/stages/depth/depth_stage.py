# import cv2
# import numpy as np
# import os
# import torch
# import tempfile
# from pathlib import Path
# import sys
# import shutil
# import time
# import psutil

# # Add Depth-Anything-V2 to Python path
# DEPTH_ANYTHING_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "third_party" / "Depth-Anything-V2")
# sys.path.insert(0, DEPTH_ANYTHING_PATH)
# from depth_anything_v2.dpt import DepthAnythingV2

# class DepthStage:
#     def __init__(self, model_path, model_type='vitl', input_size=518, batch_size=8):
#         self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
#         self.input_size = input_size
#         self.batch_size = batch_size
        
#         # Model configurations
#         self.model_configs = {
#             'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
#             'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#             'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
#             'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
#         }
        
#         # Initialize model
#         self.depth_anything = DepthAnythingV2(**self.model_configs[model_type])
#         if not os.path.exists(model_path):
#             raise ValueError(f"Checkpoint not found at {model_path}")
#         self.depth_anything.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.depth_anything = self.depth_anything.to(self.device).eval()

#     def process(self, input_video, output_dir):
#         """
#         Process video and generate depth maps
#         """
#         start_time = time.time()
#         initial_ram = psutil.Process().memory_info().rss
#         if torch.cuda.is_available():
#             torch.cuda.reset_peak_memory_stats()

#         # Check and recreate output directory
#         if os.path.exists(output_dir):
#             print(f"Output directory '{output_dir}' already exists. Deleting and recreating it.")
#             shutil.rmtree(output_dir)
#         os.makedirs(output_dir, exist_ok=True)

#         # Create temporary directory for frames
#         temp_frames_dir = os.path.join(output_dir, "temporal_frames")
#         os.makedirs(temp_frames_dir, exist_ok=True)
        
#         # Extract frames from video
#         print(f"Extracting frames from video: {input_video}")
#         cap = cv2.VideoCapture(input_video)
#         frame_count = 0
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_path = os.path.join(temp_frames_dir, f'frame_{frame_count:06d}.jpg')
#             cv2.imwrite(frame_path, frame)
#             frame_count += 1
        
#         cap.release()
        
#         # Process frames in batches
#         frame_files = sorted([os.path.join(temp_frames_dir, f) for f in os.listdir(temp_frames_dir) if f.endswith('.jpg')])
        
#         print(f"\nProcessing {len(frame_files)} frames in batches of {self.batch_size}...")
#         for batch_idx in range(0, len(frame_files), self.batch_size):
#             batch_files = frame_files[batch_idx:batch_idx + self.batch_size]
#             print(f"\nProcessing batch {batch_idx//self.batch_size + 1}/{(len(frame_files)-1)//self.batch_size + 1}")
            
#             # Process each frame in the batch
#             for i, filename in enumerate(batch_files):
#                 frame_number = batch_idx + i
#                 print(f'Processing frame {frame_number+1}/{len(frame_files)}')
                
#                 raw_image = cv2.imread(filename)
#                 depth = self.depth_anything.infer_image(raw_image, self.input_size)
                
#                 # Normalize depth map
#                 depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
#                 depth_normalized = depth_normalized.astype(np.uint8)
#                 depth_normalized = 255 - depth_normalized
                
#                 # Save grayscale depth map
#                 depth_filename = os.path.join(output_dir, f'depth_{frame_number:06d}.png')
#                 cv2.imwrite(depth_filename, depth_normalized, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                
#                 # Verify single channel
#                 saved_depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
#                 if len(saved_depth.shape) == 2:
#                     print(f"Depth map {frame_number} saved correctly as single channel image")
#                 else:
#                     print(f"Warning! Depth map {frame_number} has {saved_depth.shape[-1]} channels")

#         # Cleanup and metrics
#         shutil.rmtree(temp_frames_dir)
        
#         total_time = time.time() - start_time
#         final_ram = psutil.Process().memory_info().rss
#         ram_usage = (final_ram - initial_ram) / (1024**2)  # en MB
        
#         print("\nPerformance Metrics:")
#         print(f"Total processing time: {total_time:.2f} seconds")
#         print(f"RAM Usage: {ram_usage:.2f} MB")
        
#         if torch.cuda.is_available():
#             gpu_peak = torch.cuda.max_memory_allocated() / (1024**3)  # en GB
#             print(f"GPU Peak Memory: {gpu_peak:.2f} GB")
        
#         return output_dir

# import cv2
# import numpy as np
# import os
# import torch
# import tempfile
# from pathlib import Path
# import sys
# import shutil

# # Add Depth-Anything-V2 to Python path
# DEPTH_ANYTHING_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "third_party" / "Depth-Anything-V2")
# sys.path.insert(0, DEPTH_ANYTHING_PATH)
# from depth_anything_v2.dpt import DepthAnythingV2



# class DepthStage:
#     def __init__(self, model_path, model_type='vitl', input_size=518):
#         self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
#         self.input_size = input_size
        
#         # Model configurations
#         self.model_configs = {
#             'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
#             'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#             'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
#             'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
#         }
        
#         # Initialize model
#         self.depth_anything = DepthAnythingV2(**self.model_configs[model_type])
#         if not os.path.exists(model_path):
#             raise ValueError(f"Checkpoint not found at {model_path}")
#         self.depth_anything.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.depth_anything = self.depth_anything.to(self.device).eval()

#     def process(self, input_video, output_dir):
#         """
#         Process video and generate depth maps
#         """

#         # Check and recreate output directory
#         if os.path.exists(output_dir):
#             print(f"Output directory '{output_dir}' already exists. Deleting and recreating it.")
#             shutil.rmtree(output_dir)
#         os.makedirs(output_dir, exist_ok=True)

        
#         # Create temporary directory for frames
#         temp_frames_dir = os.path.join(output_dir, "temporal_frames")
#         os.makedirs(temp_frames_dir, exist_ok=True)
        
#         # Extract frames from video
#         print(f"Extracting frames from video: {input_video}")
#         cap = cv2.VideoCapture(input_video)
#         frame_count = 0
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_path = os.path.join(temp_frames_dir, f'frame_{frame_count:06d}.jpg')
#             cv2.imwrite(frame_path, frame)
#             frame_count += 1
        
#         cap.release()
        
#         # Process frames
#         frame_files = sorted([os.path.join(temp_frames_dir, f) for f in os.listdir(temp_frames_dir) if f.endswith('.jpg')])
        
#         print(f"Processing {len(frame_files)} frames...")
#         for k, filename in enumerate(frame_files):
#             print(f'Processing frame {k+1}/{len(frame_files)}')
            
#             raw_image = cv2.imread(filename)
#             depth = self.depth_anything.infer_image(raw_image, self.input_size)
            
#             # Normalize depth map
#             depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
#             depth_normalized = depth_normalized.astype(np.uint8)
#             depth_normalized = 255 - depth_normalized
            
#             # Save grayscale depth map
#             depth_filename = os.path.join(output_dir, f'depth_{k:06d}.png')
#             cv2.imwrite(depth_filename, depth_normalized, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            
#             # Verify single channel
#             saved_depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
#             if len(saved_depth.shape) == 2:
#                 print(f"Depth map {k} saved correctly as single channel image")
#             else:
#                 print(f"Warning! Depth map {k} has {saved_depth.shape[-1]} channels")
        
#         # Clean up temporary directory
#         shutil.rmtree(temp_frames_dir)
        
#         return output_dir

import cv2
import numpy as np
import os
import torch
import tempfile
from pathlib import Path
import sys
import shutil
import time
import psutil

# Add Depth-Anything-V2 to Python path
DEPTH_ANYTHING_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "third_party" / "Depth-Anything-V2")
sys.path.insert(0, DEPTH_ANYTHING_PATH)
from depth_anything_v2.dpt import DepthAnythingV2

class DepthStage:
    def __init__(self, model_path, model_type='vitl', input_size=518):
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.input_size = input_size
        
        # Model configurations
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        # Initialize model
        self.depth_anything = DepthAnythingV2(**self.model_configs[model_type])
        if not os.path.exists(model_path):
            raise ValueError(f"Checkpoint not found at {model_path}")
        self.depth_anything.load_state_dict(torch.load(model_path, map_location=self.device))
        self.depth_anything = self.depth_anything.to(self.device).eval()

    def process(self, input_video, output_dir):
        """
        Process video and generate depth maps
        """
        start_time = time.time()
        initial_ram = psutil.Process().memory_info().rss
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Check and recreate output directory
        if os.path.exists(output_dir):
            print(f"Output directory '{output_dir}' already exists. Deleting and recreating it.")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Create temporary directory for frames
        temp_frames_dir = os.path.join(output_dir, "temporal_frames")
        os.makedirs(temp_frames_dir, exist_ok=True)
        
        # Extract frames from video
        print(f"Extracting frames from video: {input_video}")
        cap = cv2.VideoCapture(input_video)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = os.path.join(temp_frames_dir, f'frame_{frame_count:06d}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        
        cap.release()
        
        # Process frames
        frame_files = sorted([os.path.join(temp_frames_dir, f) for f in os.listdir(temp_frames_dir) if f.endswith('.jpg')])
        
        print(f"Processing {len(frame_files)} frames...")
        for k, filename in enumerate(frame_files):
            print(f'Processing frame {k+1}/{len(frame_files)}')
            
            raw_image = cv2.imread(filename)
            depth = self.depth_anything.infer_image(raw_image, self.input_size)
            
            # Normalize depth map
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth_normalized = depth_normalized.astype(np.uint8)
            depth_normalized = 255 - depth_normalized
            
            # Save grayscale depth map
            depth_filename = os.path.join(output_dir, f'depth_{k:06d}.png')
            cv2.imwrite(depth_filename, depth_normalized, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            
            # Verify single channel
            saved_depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
            if len(saved_depth.shape) == 2:
                print(f"Depth map {k} saved correctly as single channel image")
            else:
                print(f"Warning! Depth map {k} has {saved_depth.shape[-1]} channels")
        
        # Clean up temporary directory
        shutil.rmtree(temp_frames_dir)
        
        # Performance metrics
        total_time = time.time() - start_time
        final_ram = psutil.Process().memory_info().rss
        ram_usage = (final_ram - initial_ram) / (1024**2)  # Convert to MB
        
        print("\nPerformance Metrics:")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"RAM Usage: {ram_usage:.2f} MB")
        
        if torch.cuda.is_available():
            gpu_peak = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
            print(f"GPU Peak Memory: {gpu_peak:.2f} GB")
        else:
            print("GPU Peak Memory: Not available (using CPU)")
        
        return output_dir