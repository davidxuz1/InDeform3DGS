import os
import shutil
import tempfile
import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2_video_predictor

class SegmentationStage:
    def __init__(self, model_path, batch_size=50, dilatation_factor=1.0, save_masks=False):
        self.model_path = model_path
        self.batch_size = batch_size
        self.dilatation_factor = dilatation_factor
        self.save_masks = save_masks
        self.frame_boxes = None

    def parse_bbox_line(self, line):
        boxes = []
        parts = line.split(": ")
        if len(parts) < 2 or not parts[1].strip():
            return boxes
        bbox_parts = parts[1].strip()
        bbox_strings = bbox_parts.split("] [")
        for bbox_str in bbox_strings:
            bbox_str = bbox_str.replace("[", "").replace("]", "")
            if bbox_str:
                coords = list(map(int, bbox_str.split(",")))
                boxes.append(coords)
        return boxes

    def read_bbox_file(self, file_path):        
        frame_boxes = {}
        with open(file_path, 'r') as f:
            for line in f:
                frame_num = int(line.split("_")[1].split(":")[0])
                boxes = self.parse_bbox_line(line)
                if boxes:
                    frame_boxes[frame_num] = boxes
        return frame_boxes

    def dilate_mask(self, mask, dilatation_factor):
        kernel_size = int(round(dilatation_factor * 2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(mask, kernel, iterations=1)

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection / float(area1 + area2 - intersection)

    def combine_and_reduce_boxes(self, box1, box2, factor=1.05):
        x1_min = min(box1[0], box2[0])
        y1_min = min(box1[1], box2[1])
        x2_max = max(box1[2], box2[2])
        y2_max = max(box1[3], box2[3])

        center_x = (x1_min + x2_max) / 2.0
        center_y = (y1_min + y2_max) / 2.0

        width = x2_max - x1_min
        height = y2_max - y1_min

        new_width = width / factor
        new_height = height / factor

        new_x1 = int(center_x - new_width / 2)
        new_y1 = int(center_y - new_height / 2)
        new_x2 = int(center_x + new_width / 2)
        new_y2 = int(center_y + new_height / 2)

        return [new_x1, new_y1, new_x2, new_y2]
    def check_forward_consistency(self, frame_boxes, current_frame, lookforward=4):
        frames_to_check = [current_frame + i for i in range(lookforward)]
        if not all(f in frame_boxes for f in frames_to_check):
            return None

        boxes_sequence = [frame_boxes[f] for f in frames_to_check]
        consistent_boxes = []
        all_boxes_with_frame = []
        for i, boxes in enumerate(boxes_sequence):
            for box in boxes:
                all_boxes_with_frame.append((i, box))

        used = [False] * len(all_boxes_with_frame)
        iou_cache = {}  

        for idx, (f_idx, box) in enumerate(all_boxes_with_frame):
            if used[idx]:
                continue

            cluster_boxes = [(f_idx, box)]
            used[idx] = True

            for jdx, (other_f_idx, other_box) in enumerate(all_boxes_with_frame):
                if used[jdx]:
                    continue

                cache_key = (idx, jdx) if idx < jdx else (jdx, idx)
                if cache_key in iou_cache:
                    iou = iou_cache[cache_key]
                else:
                    iou = self.calculate_iou(box, other_box)
                    iou_cache[cache_key] = iou

                if iou > 0.7:
                    cluster_boxes.append((other_f_idx, other_box))
                    used[jdx] = True

            distinct_frames = set([c[0] for c in cluster_boxes])
            if len(distinct_frames) >= 3:
                cluster_boxes_sorted = sorted(cluster_boxes, key=lambda x: x[0])
                representative_box = cluster_boxes_sorted[0][1]
                if representative_box not in consistent_boxes:
                    consistent_boxes.append(representative_box)

        if not consistent_boxes:
            return None

        merged = True
        while merged and len(consistent_boxes) > 1:
            merged = False
            new_consistent_boxes = []
            used_indices = set()

            for i in range(len(consistent_boxes)):
                if i in used_indices:
                    continue
                box_a = consistent_boxes[i]
                merged_with_some = False
                for j in range(i + 1, len(consistent_boxes)):
                    if j in used_indices:
                        continue

                    cache_key = (i, j) if i < j else (j, i)
                    if cache_key in iou_cache:
                        iou = iou_cache[cache_key]
                    else:
                        iou = self.calculate_iou(box_a, consistent_boxes[j])
                        iou_cache[cache_key] = iou

                    if iou > 0.2:
                        combined_box = self.combine_and_reduce_boxes(box_a, consistent_boxes[j], factor=1.05)
                        new_consistent_boxes.append(combined_box)
                        used_indices.add(i)
                        used_indices.add(j)
                        merged = True
                        merged_with_some = True
                        break

                if not merged_with_some and i not in used_indices:
                    new_consistent_boxes.append(box_a)

            consistent_boxes = new_consistent_boxes

        return consistent_boxes if consistent_boxes else None
    def process(self, video_input, bbox_file, output_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        frame_boxes = self.read_bbox_file(bbox_file)
        if not frame_boxes:
            raise ValueError("No valid bounding boxes found in the file")

        print("Initializing SAM2 model...")
        predictor = build_sam2_video_predictor(
            "sam2_hiera_t.yaml",
            self.model_path,
            device=device
        )

        all_binary_masks = []
        if self.save_masks:
            masks_dir = os.path.join(os.path.dirname(output_path), 'binary_masks')
            os.makedirs(masks_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            original_frames_dir = os.path.join(temp_dir, 'original_frames')
            batch_frames_dir = os.path.join(temp_dir, 'batch_frames')
            os.makedirs(original_frames_dir, exist_ok=True)
            os.makedirs(batch_frames_dir, exist_ok=True)

            print(f"Extracting frames from video: {video_input}")
            os.system(f"ffmpeg -i {video_input} -q:v 2 -start_number 0 {original_frames_dir}/%05d.jpg")

            frame_names = sorted([f for f in os.listdir(original_frames_dir) if f.endswith('.jpg')])
            total_frames = len(frame_names)
            print(f"Total frames to process: {total_frames}")
            
            frames = [cv2.imread(os.path.join(original_frames_dir, frame_name)) for frame_name in frame_names]
            
            first_frame = cv2.imread(os.path.join(original_frames_dir, frame_names[0]))
            frame_height, frame_width, _ = first_frame.shape
            print(f"Frame dimensions: {frame_width}x{frame_height}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

            try:
                batch_start = 0
                while batch_start < total_frames:
                    batch_frame_num = batch_start + 1

                    forward_consistent_boxes = self.check_forward_consistency(frame_boxes, batch_frame_num)
                    
                    if forward_consistent_boxes:
                        initial_boxes = forward_consistent_boxes
                        batch_end = min(batch_start + self.batch_size, total_frames)
                        print(f"Found consistent boxes in frames {batch_frame_num}-{batch_frame_num+2}: {initial_boxes}")
                        print(f"\nProcessing batch {batch_start}-{batch_end}")
                    else:
                        batch_start += 1
                        print(f"No consistent boxes found for frames {batch_frame_num}-{batch_frame_num+2}, moving to next frame")
                        continue

                    batch_frames = frame_names[batch_start:batch_end]
                    for frame_name in os.listdir(batch_frames_dir):
                        os.remove(os.path.join(batch_frames_dir, frame_name))
                    
                    for frame_name in batch_frames:
                        shutil.copy(
                            os.path.join(original_frames_dir, frame_name),
                            os.path.join(batch_frames_dir, frame_name)
                        )

                    try:
                        inference_state = predictor.init_state(video_path=batch_frames_dir)

                        for obj_id, box in enumerate(initial_boxes, start=1):
                            _, _, _ = predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=0,
                                obj_id=obj_id,
                                box=np.array(box, dtype=np.float32)
                            )

                        print("Propagating segmentation...")
                        video_segments = {}
                        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                            video_segments[out_frame_idx] = {
                                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                                for i, out_obj_id in enumerate(out_obj_ids)
                            }

                        for rel_frame_idx, frame_name in enumerate(batch_frames):
                            frame = cv2.imread(os.path.join(original_frames_dir, frame_name))
                            frame = frames[batch_start + rel_frame_idx]
                            if frame is None:
                                print(f"Warning: Failed to read frame {frame_name}")
                                continue

                            combined_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

                            if rel_frame_idx in video_segments:
                                for out_obj_id, mask in video_segments[rel_frame_idx].items():
                                    if mask is None or mask.size == 0:
                                        continue

                                    mask_uint8 = (mask * 255).astype(np.uint8)
                                    if mask_uint8.ndim == 3 and mask_uint8.shape[0] == 1:
                                        mask_uint8 = np.squeeze(mask_uint8, axis=0)

                                    if mask_uint8.shape[0] == 0 or mask_uint8.shape[1] == 0:
                                        continue

                                    try:
                                        resized_mask = cv2.resize(mask_uint8, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                                        dilated_mask = self.dilate_mask(resized_mask, self.dilatation_factor)
                                        combined_mask = np.logical_or(combined_mask, dilated_mask > 0)

                                        mask_overlay = cv2.merge([dilated_mask, dilated_mask, dilated_mask])
                                        mask_color = np.array([0, 255, 0], dtype=np.uint8) if out_obj_id % 2 == 0 else np.array([255, 0, 0], dtype=np.uint8)
                                        mask_overlay = (mask_overlay > 0) * mask_color
                                        frame = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)

                                    except cv2.error as e:
                                        print(f"Error processing mask for frame {frame_name}: {e}")
                                        continue

                            if self.save_masks:
                                mask_filename = f"mask_{batch_start + rel_frame_idx:05d}.png"
                                mask_path = os.path.join(masks_dir, mask_filename)
                                binary_mask = (combined_mask * 255).astype(np.uint8)
                                cv2.imwrite(mask_path, binary_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

                            all_binary_masks.append(combined_mask.astype(np.bool_))

                            if rel_frame_idx == 0:
                                for box in initial_boxes:
                                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
                            
                            frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
                            out.write(frame)

                    except RuntimeError as e:
                        print(f"Error processing batch {batch_start}-{batch_end}: {e}")

                    batch_start = batch_end

                out.release()
                masks_output_path = output_path.rsplit('.', 1)[0] + '_masks.npy'
                np.save(masks_output_path, np.array(all_binary_masks))
                print(f"Segmented video saved to {output_path}")
                print(f"Binary masks saved to {masks_output_path}")

            except Exception as e:
                print(f"An error occurred during video processing: {e}")
                raise

        return output_path



# import os
# import shutil
# import tempfile
# import numpy as np
# import torch
# import cv2
# from sam2.build_sam import build_sam2_video_predictor

# class SegmentationStage:
#     def __init__(self, model_path, batch_size=50, dilatation_factor=1.0, save_masks=False):
#         self.model_path = model_path
#         self.batch_size = batch_size
#         self.dilatation_factor = dilatation_factor
#         self.save_masks = save_masks
#         self.frame_boxes = None

#     def parse_bbox_line(self, line):
#         boxes = []
#         parts = line.split(": ")
#         if len(parts) < 2 or not parts[1].strip():
#             return boxes
#         bbox_parts = parts[1].strip()
#         bbox_strings = bbox_parts.split("] [")
#         for bbox_str in bbox_strings:
#             bbox_str = bbox_str.replace("[", "").replace("]", "")
#             if bbox_str:
#                 coords = list(map(int, bbox_str.split(",")))
#                 boxes.append(coords)
#         return boxes

#     def read_bbox_file(self, file_path):        
#         frame_boxes = {}
#         with open(file_path, 'r') as f:
#             for line in f:
#                 frame_num = int(line.split("_")[1].split(":")[0])
#                 boxes = self.parse_bbox_line(line)
#                 if boxes:
#                     frame_boxes[frame_num] = boxes
#         return frame_boxes

#     def dilate_mask(self, mask, dilatation_factor):
#         kernel_size = int(round(dilatation_factor * 2))
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
#         return cv2.dilate(mask, kernel, iterations=1)

#     def calculate_iou(self, box1, box2):
#         x1 = max(box1[0], box2[0])
#         y1 = max(box1[1], box2[1])
#         x2 = min(box1[2], box2[2])
#         y2 = min(box1[3], box2[3])
#         intersection = max(0, x2 - x1) * max(0, y2 - y1)
#         area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#         area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#         return intersection / float(area1 + area2 - intersection)

#     def combine_and_reduce_boxes(self, box1, box2, factor=1.05):
#         """
#         Combina dos boxes tomando la unión que los cubra a ambos y luego
#         reduce el tamaño del box resultante por un factor dado.

#         box formato: [x1, y1, x2, y2]
#         factor > 1: el box se vuelve más pequeño
#         """
#         x1_min = min(box1[0], box2[0])
#         y1_min = min(box1[1], box2[1])
#         x2_max = max(box1[2], box2[2])
#         y2_max = max(box1[3], box2[3])

#         # Calcular el centro del box combinado
#         center_x = (x1_min + x2_max) / 2.0
#         center_y = (y1_min + y2_max) / 2.0

#         width = x2_max - x1_min
#         height = y2_max - y1_min

#         # Reducir el tamaño del box
#         new_width = width / factor
#         new_height = height / factor

#         new_x1 = int(center_x - new_width / 2)
#         new_y1 = int(center_y - new_height / 2)
#         new_x2 = int(center_x + new_width / 2)
#         new_y2 = int(center_y + new_height / 2)

#         return [new_x1, new_y1, new_x2, new_y2]

#     def check_forward_consistency(self, frame_boxes, current_frame, lookforward=4):
#         """
#         Verifica la consistencia de los boxes en los siguientes frames.
#         Se consideran 4 frames en total: current_frame, current_frame+1, current_frame+2, current_frame+3.
#         Un box se considera consistente si aparece con IoU > 0.7 en al menos 3 de estos 4 frames.

#         Luego, como segundo filtro, si entre los boxes consistentes (ya filtrados) 
#         hay dos con IoU > 0.6, se crea un nuevo box que es la unión de ambos y 
#         luego se reduce su tamaño por un factor de 1.5.
#         """
#         frames_to_check = [current_frame + i for i in range(lookforward)]
#         # Verificar que todos los frames existan, si no devolver None
#         if not all(f in frame_boxes for f in frames_to_check):
#             return None

#         # Obtener la lista de boxes para cada frame
#         boxes_sequence = [frame_boxes[f] for f in frames_to_check]

#         # Queremos encontrar boxes que se correspondan a la misma "instancia" a lo largo de los frames
#         # y que aparezcan en al menos 3 de estos 4 frames.
#         consistent_boxes = []

#         # Combinar todos los boxes en una sola lista con referencia a qué frame pertenecen
#         all_boxes_with_frame = []
#         for i, boxes in enumerate(boxes_sequence):
#             for box in boxes:
#                 all_boxes_with_frame.append((i, box))  # i = 0..3

#         used = [False] * len(all_boxes_with_frame)

#         # Primer filtrado: encontrar clusters con IoU > 0.7 en al menos 3 de 4 frames
#         for idx, (f_idx, box) in enumerate(all_boxes_with_frame):
#             if used[idx]:
#                 continue

#             cluster_boxes = [(f_idx, box)]
#             used[idx] = True

#             # Buscar boxes similares
#             for jdx, (other_f_idx, other_box) in enumerate(all_boxes_with_frame):
#                 if used[jdx]:
#                     continue
#                 for (_, c_box) in cluster_boxes:
#                     if self.calculate_iou(c_box, other_box) > 0.7:
#                         cluster_boxes.append((other_f_idx, other_box))
#                         used[jdx] = True
#                         break

#             distinct_frames = set([c[0] for c in cluster_boxes])
#             if len(distinct_frames) >= 3:
#                 cluster_boxes_sorted = sorted(cluster_boxes, key=lambda x: x[0])
#                 representative_box = cluster_boxes_sorted[0][1]
#                 if representative_box not in consistent_boxes:
#                     consistent_boxes.append(representative_box)

#         if not consistent_boxes:
#             return None

#         # Segundo filtrado: entre los consistent_boxes, si IoU > 0.2, combinar y reducir
#         merged = True
#         while merged and len(consistent_boxes) > 1:
#             merged = False
#             new_consistent_boxes = []
#             used_indices = set()

#             for i in range(len(consistent_boxes)):
#                 if i in used_indices:
#                     continue
#                 box_a = consistent_boxes[i]
#                 merged_with_some = False
#                 for j in range(i+1, len(consistent_boxes)):
#                     if j in used_indices:
#                         continue
#                     box_b = consistent_boxes[j]
#                     iou = self.calculate_iou(box_a, box_b)
#                     print(f"IoU between {box_a} and {box_b}: {iou}")
#                     if iou > 0.2:
#                         combined_box = self.combine_and_reduce_boxes(box_a, box_b, factor=1.05)
#                         new_consistent_boxes.append(combined_box)
#                         used_indices.add(i)
#                         used_indices.add(j)
#                         merged = True
#                         merged_with_some = True
#                         break
#                 if not merged_with_some and i not in used_indices:
#                     new_consistent_boxes.append(box_a)
            
#             consistent_boxes = new_consistent_boxes

#         return consistent_boxes if consistent_boxes else None

#     def check_forward_consistency_2(self, frame_boxes, current_frame, lookforward=4):
#         """
#         Verifica la consistencia de los boxes en los siguientes frames.
#         Se consideran 4 frames en total: current_frame, current_frame+1, current_frame+2, current_frame+3.
#         Un box se considera consistente si aparece con IoU > 0.7 en al menos 3 de los 4 frames.

#         Devuelve una lista de boxes consistentes o None si no encuentra ninguno.
#         """
#         frames_to_check = [current_frame + i for i in range(lookforward)]
#         # Verificar que todos los frames existan, si no devolver None
#         if not all(f in frame_boxes for f in frames_to_check):
#             return None

#         # Obtener la lista de boxes para cada frame
#         boxes_sequence = [frame_boxes[f] for f in frames_to_check]

#         consistent_boxes = []
#         all_boxes_with_frame = []
#         for i, boxes in enumerate(boxes_sequence):
#             for box in boxes:
#                 all_boxes_with_frame.append((i, box))

#         used = [False] * len(all_boxes_with_frame)

#         for idx, (f_idx, box) in enumerate(all_boxes_with_frame):
#             if used[idx]:
#                 continue

#             cluster_boxes = [(f_idx, box)]
#             used[idx] = True

#             for jdx, (other_f_idx, other_box) in enumerate(all_boxes_with_frame):
#                 if used[jdx]:
#                     continue
#                 for (_, c_box) in cluster_boxes:
#                     if self.calculate_iou(c_box, other_box) > 0.7:
#                         cluster_boxes.append((other_f_idx, other_box))
#                         used[jdx] = True
#                         break

#             distinct_frames = set([c[0] for c in cluster_boxes])
#             if len(distinct_frames) >= 3:
#                 cluster_boxes_sorted = sorted(cluster_boxes, key=lambda x: x[0])
#                 representative_box = cluster_boxes_sorted[0][1]
#                 if representative_box not in consistent_boxes:
#                     consistent_boxes.append(representative_box)

#         return consistent_boxes if consistent_boxes else None

#     def process(self, video_input, bbox_file, output_path):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using device: {device}")

#         frame_boxes = self.read_bbox_file(bbox_file)
#         if not frame_boxes:
#             raise ValueError("No valid bounding boxes found in the file")

#         print("Initializing SAM2 model...")
#         predictor = build_sam2_video_predictor(
#             "sam2_hiera_t.yaml",
#             self.model_path,
#             device=device
#         )

#         all_binary_masks = []
#         if self.save_masks:
#             masks_dir = os.path.join(os.path.dirname(output_path), 'binary_masks')
#             os.makedirs(masks_dir, exist_ok=True)

#         with tempfile.TemporaryDirectory() as temp_dir:
#             original_frames_dir = os.path.join(temp_dir, 'original_frames')
#             batch_frames_dir = os.path.join(temp_dir, 'batch_frames')
#             os.makedirs(original_frames_dir, exist_ok=True)
#             os.makedirs(batch_frames_dir, exist_ok=True)

#             print(f"Extracting frames from video: {video_input}")
#             os.system(f"ffmpeg -i {video_input} -q:v 2 -start_number 0 {original_frames_dir}/%05d.jpg")

#             frame_names = sorted([f for f in os.listdir(original_frames_dir) if f.endswith('.jpg')])
#             total_frames = len(frame_names)
#             print(f"Total frames to process: {total_frames}")

#             first_frame = cv2.imread(os.path.join(original_frames_dir, frame_names[0]))
#             frame_height, frame_width, _ = first_frame.shape
#             print(f"Frame dimensions: {frame_width}x{frame_height}")

#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

#             try:
#                 batch_start = 0
#                 while batch_start < total_frames:
#                     batch_frame_num = batch_start + 1

#                     # Verificar consistencia hacia adelante con segundo filtro
#                     forward_consistent_boxes = self.check_forward_consistency(frame_boxes, batch_frame_num)
                    
#                     if forward_consistent_boxes:
#                         initial_boxes = forward_consistent_boxes
#                         batch_end = min(batch_start + self.batch_size, total_frames)
#                         print(f"Found consistent boxes in frames {batch_frame_num}-{batch_frame_num+2}: {initial_boxes}")
#                         print(f"\nProcessing batch {batch_start}-{batch_end}")
#                     else:
#                         # Si no hay boxes consistentes, intentar con el siguiente frame
#                         batch_start += 1
#                         print(f"No consistent boxes found for frames {batch_frame_num}-{batch_frame_num+2}, moving to next frame")
#                         continue

#                     batch_frames = frame_names[batch_start:batch_end]
#                     for frame_name in os.listdir(batch_frames_dir):
#                         os.remove(os.path.join(batch_frames_dir, frame_name))
                    
#                     for frame_name in batch_frames:
#                         shutil.copy(
#                             os.path.join(original_frames_dir, frame_name),
#                             os.path.join(batch_frames_dir, frame_name)
#                         )

#                     try:
#                         inference_state = predictor.init_state(video_path=batch_frames_dir)

#                         for obj_id, box in enumerate(initial_boxes, start=1):
#                             _, _, _ = predictor.add_new_points_or_box(
#                                 inference_state=inference_state,
#                                 frame_idx=0,
#                                 obj_id=obj_id,
#                                 box=np.array(box, dtype=np.float32)
#                             )

#                         print("Propagating segmentation...")
#                         video_segments = {}
#                         for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#                             video_segments[out_frame_idx] = {
#                                 out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#                                 for i, out_obj_id in enumerate(out_obj_ids)
#                             }

#                         for rel_frame_idx, frame_name in enumerate(batch_frames):
#                             frame = cv2.imread(os.path.join(original_frames_dir, frame_name))
#                             if frame is None:
#                                 print(f"Warning: Failed to read frame {frame_name}")
#                                 continue

#                             combined_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

#                             if rel_frame_idx in video_segments:
#                                 for out_obj_id, mask in video_segments[rel_frame_idx].items():
#                                     if mask is None or mask.size == 0:
#                                         continue

#                                     mask_uint8 = (mask * 255).astype(np.uint8)
#                                     if mask_uint8.ndim == 3 and mask_uint8.shape[0] == 1:
#                                         mask_uint8 = np.squeeze(mask_uint8, axis=0)

#                                     if mask_uint8.shape[0] == 0 or mask_uint8.shape[1] == 0:
#                                         continue

#                                     try:
#                                         resized_mask = cv2.resize(mask_uint8, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
#                                         dilated_mask = self.dilate_mask(resized_mask, self.dilatation_factor)
#                                         combined_mask = np.logical_or(combined_mask, dilated_mask > 0)

#                                         mask_overlay = cv2.merge([dilated_mask, dilated_mask, dilated_mask])
#                                         mask_color = np.array([0, 255, 0], dtype=np.uint8) if out_obj_id % 2 == 0 else np.array([255, 0, 0], dtype=np.uint8)
#                                         mask_overlay = (mask_overlay > 0) * mask_color
#                                         frame = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)

#                                     except cv2.error as e:
#                                         print(f"Error processing mask for frame {frame_name}: {e}")
#                                         continue

#                             if self.save_masks:
#                                 mask_filename = f"mask_{batch_start + rel_frame_idx:05d}.png"
#                                 mask_path = os.path.join(masks_dir, mask_filename)
#                                 binary_mask = (combined_mask * 255).astype(np.uint8)
#                                 cv2.imwrite(mask_path, binary_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

#                             all_binary_masks.append(combined_mask.astype(np.bool_))

#                             # Dibujar los bounding boxes finales (initial_boxes) en amarillo
#                             if rel_frame_idx == 0:
#                                 for box in initial_boxes:
#                                     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
                            
#                             # Ajuste opcional de brillo/contraste
#                             frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
#                             out.write(frame)

#                     except RuntimeError as e:
#                         print(f"Error processing batch {batch_start}-{batch_end}: {e}")

#                     batch_start = batch_end

#                 out.release()
#                 masks_output_path = output_path.rsplit('.', 1)[0] + '_masks.npy'
#                 np.save(masks_output_path, np.array(all_binary_masks))
#                 print(f"Segmented video saved to {output_path}")
#                 print(f"Binary masks saved to {masks_output_path}")

#             except Exception as e:
#                 print(f"An error occurred during video processing: {e}")
#                 raise

#         return output_path
