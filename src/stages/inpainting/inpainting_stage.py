import torch
import numpy as np
import cv2
import sys
import imageio
import importlib
import os
from pathlib import Path
from typing import List
from PIL import Image
from torchvision import transforms
import argparse

import time
import psutil

STTN_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "third_party" / "STTN")
sys.path.insert(0, STTN_PATH)
from core.utils import Stack, ToTorchFormatTensor

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()
])

class VideoInpainter:
    def __init__(self, vi_ckpt: str, device: str = None, width: int = 432, height: int = 240,
                 neighbor_stride: int = 5, ref_length: int = 10, anchor_stride: int = 10,
                 enable_blending: bool = True):
        self.enable_blending = enable_blending
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.inpainter = self._build_sttn_model(vi_ckpt)
        self.base_w, self.base_h = width, height
        self.neighbor_stride = neighbor_stride
        self.ref_length = ref_length
        self.anchor_stride = anchor_stride
        
        self.w = self.base_w
        self.h = self.base_h

    def _build_sttn_model(self, ckpt_path: str, model_type: str = "sttn"):
        sys.path.insert(0, STTN_PATH)
        net = importlib.import_module(f'model.{model_type}')
        model = net.InpaintGenerator().to(self.device)
        data = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(data['netG'])
        model.eval()
        return model

    @staticmethod
    def _get_ref_index(neighbor_ids: List[int], length: int, ref_length: int) -> List[int]:
        return [i for i in range(0, length, ref_length) if i not in neighbor_ids]



    @torch.no_grad()
    def inpaint_video(self, frames: List[Image.Image], masks: List[Image.Image]) -> List[Image.Image]:
        video_length = len(frames)

        ori_w, ori_h = frames[0].size

        # Downsampling:
        if ori_w > 1280 or ori_h > 720:
            #self.w = self.base_w // 2
            #self.h = self.base_h // 2
            self.w = self.base_w
            self.h = self.base_h
        else:
            self.w = self.base_w
            self.h = self.base_h
            # print('ori_w, ori_h: ', ori_w, ori_h)
            # def make_divisible(val, divisor):
            #     return (val + divisor - 1) // divisor * divisor
            # self.w = make_divisible(ori_w, 16)
            # self.h = make_divisible(ori_h, 16)
            # print('w, h: ', self.w, self.h)


        resize_transform = transforms.Compose([
            transforms.Resize((self.h, self.w), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        
        resize_mask_transform = transforms.Compose([
            transforms.Resize((self.h, self.w), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        anchor_frames = [i for i in range(0, video_length, self.anchor_stride)]




        # without batch
        # start_time = time.time()
        # initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # processed_frames = []
        # processed_masks = []

        # for frame, mask in zip(frames, masks):
        #     resized_frame = resize_transform(frame).unsqueeze(0)  # Añadir dimensión de batch
        #     resized_mask = resize_mask_transform(mask).unsqueeze(0)

        #     # Normalizar y mover a device
        #     resized_frame = (resized_frame * 2.0 - 1.0).to(self.device)
        #     resized_mask = resized_mask.to(self.device)

        #     processed_frames.append(resized_frame)
        #     processed_masks.append(resized_mask)

        # # Concatenar todos los frames procesados
        # resized_frames_tensor = torch.cat(processed_frames, dim=0)
        # resized_masks_tensor = torch.cat(processed_masks, dim=0)

        # end_time = time.time()
        # final_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # print("\n--- Métricas de procesamiento individual ---")
        # print(f"Tiempo total: {end_time - start_time:.3f} segundos")
        # print(f"Memoria RAM usada: {final_memory - initial_memory:.2f} MB")
        # if torch.cuda.is_available():
        #     print(f"Pico máximo de memoria GPU: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")


        # with batch
        # batch_start_time = time.time()
        # initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # resized_frames_tensor = torch.stack([resize_transform(f) for f in frames], dim=0)
        # resized_masks_tensor = torch.stack([resize_mask_transform(m) for m in masks], dim=0)

        # resized_frames_tensor = (resized_frames_tensor * 2.0 - 1.0).to(self.device)
        # resized_masks_tensor = resized_masks_tensor.to(self.device)

        # batch_end_time = time.time()
        # final_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # print("\n--- Métricas de procesamiento por lotes ---")
        # print(f"Tiempo total: {batch_end_time - batch_start_time:.3f} segundos")
        # print(f"Memoria RAM usada: {final_memory - initial_memory:.2f} MB")
        # if torch.cuda.is_available():
        #     print(f"Pico máximo de memoria GPU: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")



        resized_frames_tensor = torch.stack([resize_transform(f) for f in frames], dim=0)  # [T, C, H, W]
        resized_masks_tensor = torch.stack([resize_mask_transform(m) for m in masks], dim=0)  # [T, 1, H, W]


        resized_frames_tensor = resized_frames_tensor * 2.0 - 1.0

        resized_frames_tensor = resized_frames_tensor.to(self.device)  # [T, C, H, W]
        resized_masks_tensor = resized_masks_tensor.to(self.device)    # [T, 1, H, W]


        feats = (resized_frames_tensor * (1 - resized_masks_tensor.float()))
        feats = self.inpainter.encoder(feats)  # [T, C, H', W']

        T, C, Hf, Wf = feats.size()
        feats = feats.unsqueeze(0)  # [1, T, C, H', W']

        comp_frames = [None] * video_length

        for f in range(0, video_length, self.neighbor_stride):
            neighbor_ids = list(range(max(0, f - self.neighbor_stride),
                                      min(video_length, f + self.neighbor_stride + 1)))
            ref_ids = self._get_ref_index(neighbor_ids, video_length, self.ref_length)

            for af in anchor_frames:
                if af not in neighbor_ids and af not in ref_ids:
                    ref_ids.append(af)

            ref_ids = sorted(ref_ids)
            selected_ids = neighbor_ids + ref_ids

            selected_feats = feats[0, selected_ids, :, :, :]          # [N+R, C, H', W']
            selected_masks = resized_masks_tensor[selected_ids, :, :, :]  # [N+R, 1, H, W]

            pred_feat = self.inpainter.infer(selected_feats, selected_masks)  # [N+R, C, H', W']
            pred_img = self.inpainter.decoder(pred_feat[:len(neighbor_ids), :, :, :])  # [N, C, H, W]
            pred_img = ((torch.tanh(pred_img) + 1) / 2) * 255.0

            for i, idx in enumerate(neighbor_ids):
                mask_bin = (resized_masks_tensor[idx] != 0).int()  # [1, H, W]

                original_frame_255 = ((resized_frames_tensor[idx] + 1) / 2) * 255.0
                combined = pred_img[i] * mask_bin + original_frame_255 * (1 - mask_bin)

                if comp_frames[idx] is None or not self.enable_blending:
                    comp_frames[idx] = combined
                else:
                    comp_frames[idx] = (comp_frames[idx] * 0.5) + (combined * 0.5)

        final_frames = []
        for idx in range(video_length):
            # comp_frames[idx]: [C, H, W], 0-255
            result_np = comp_frames[idx].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)  # [H, W, C]
            result_pil = Image.fromarray(result_np)
            result_pil = result_pil.resize((ori_w, ori_h), Image.BICUBIC)

            original_np = np.array(frames[idx])
            mask_np = np.array(masks[idx])[..., np.newaxis] != 0
            result_np = np.array(result_pil)
            final_np = result_np * mask_np + original_np * (~mask_np)
            final_frames.append(Image.fromarray(final_np.astype(np.uint8)))


        return final_frames
    

class InpaintingStage:
    def __init__(self, vi_ckpt: str):
        self.inpainter = VideoInpainter(vi_ckpt)

    def process(self, input_video: str, input_mask: str, output_path: str):
        video = imageio.get_reader(input_video)
        fps = video.get_meta_data()['fps']
        frames = [Image.fromarray(frame) for frame in video]

        
        mask = np.load(input_mask)
        if mask.dtype == np.uint8:
            mask = mask.astype(bool)
        masks = [Image.fromarray((m * 255).astype(np.uint8)) for m in mask]

        if mask.shape[0] != len(frames):
            raise ValueError("The number of masks does not match the number of frames in the video")
        with torch.no_grad():
            inpainted_frames = self.inpainter.inpaint_video(frames, masks)

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        #imageio.mimsave(str(output_path), [np.array(f) for f in inpainted_frames], fps=fps)
        imageio.mimsave(str(output_path), [np.array(f) for f in inpainted_frames], fps=fps, macro_block_size=1)

        return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video inpainting using STTN")
    parser.add_argument('--input_video', type=str, required=True, help="Path to the input video")
    parser.add_argument('--input_mask', type=str, required=True, help="Path to the video masks (npy file)")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the generated video")
    parser.add_argument('--vi_ckpt', type=str, required=True, help="Path to the STTN model checkpoint")

    args = parser.parse_args()

    if not Path(args.input_video).exists():
        raise FileNotFoundError(f"The video file does not exist: {args.input_video}")
    if not Path(args.input_mask).exists():
        raise FileNotFoundError(f"The mask file does not exist: {args.input_mask}")
    if not Path(args.vi_ckpt).exists():
        raise FileNotFoundError(f"The model checkpoint does not exist: {args.vi_ckpt}")

    stage = InpaintingStage(vi_ckpt=args.vi_ckpt)
    print("Processing video...")
    output_video = stage.process(input_video=args.input_video,
                                  input_mask=args.input_mask,
                                  output_path=args.output_path)
    print(f"Video processed and saved at: {output_video}")

# We need dehazed_video.mp4, masks.npy, and sttn.pth in the folder







