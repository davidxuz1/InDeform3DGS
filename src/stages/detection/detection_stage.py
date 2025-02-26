import os
import cv2
import numpy as np
from ultralytics import YOLO


class SurgicalToolsDetectionStage:
    def __init__(self, model_path="models/pretrained/Surgical_Tools_Detection_Yolov11_Model/surgical_tools_detection_model.pt", 
                 threshold=0.6, dilation_factor=1.4, fixed_bbox_watermark=None):
        self.model = YOLO(model_path)
        self.threshold = threshold
        self.dilation_factor = dilation_factor
        self.fixed_bbox_watermark = fixed_bbox_watermark
        self.overlay_color = (250, 181, 180) 

    @staticmethod
    def iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter_area / float(box1_area + box2_area - inter_area)

    @staticmethod
    def non_max_suppression(boxes_confidences, iou_threshold=0.5):
        boxes_confidences = sorted(boxes_confidences, key=lambda x: x[1], reverse=True)
        filtered_boxes = []
        while boxes_confidences:
            current_box = boxes_confidences.pop(0)
            filtered_boxes.append(current_box)
            boxes_confidences = [
                (box, conf) for (box, conf) in boxes_confidences
                if SurgicalToolsDetectionStage.iou(current_box[0], box) < iou_threshold
            ]
        return filtered_boxes

    @staticmethod
    def dilate_bounding_box(box, factor, frame_width, frame_height):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        new_width = width * factor
        new_height = height * factor
        center_x = x1 + width / 2
        center_y = y1 + height / 2
        new_x1 = max(0, int(center_x - new_width / 2))
        new_y1 = max(0, int(center_y - new_height / 2))
        new_x2 = min(frame_width, int(center_x + new_width / 2))
        new_y2 = min(frame_height, int(center_y + new_height / 2))
        return new_x1, new_y1, new_x2, new_y2

    @staticmethod
    def apply_clahe(frame):
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_frame)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl_l_channel = clahe.apply(l_channel)
        merged_frame = cv2.merge((cl_l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged_frame, cv2.COLOR_LAB2BGR)

    def process(self, input_path, output_path, txt_output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

        cap = cv2.VideoCapture(input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

        with open(txt_output_path, 'w') as txt_file:
            frame_idx = 1
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                preprocessed_frame = self.apply_clahe(frame)
                results = self.model.predict(source=preprocessed_frame)

                boxes_confidences = []
                for result in results:
                    for box in result.boxes:
                        confidence = box.conf.item()
                        if confidence > self.threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            boxes_confidences.append(((x1, y1, x2, y2), confidence))

                filtered_boxes = self.non_max_suppression(boxes_confidences)

                if filtered_boxes:
                    bbox_line = " ".join([
                        f"[{dx1},{dy1},{dx2},{dy2}]" 
                        for (box, _) in filtered_boxes
                        for dx1, dy1, dx2, dy2 in [self.dilate_bounding_box(
                            box, self.dilation_factor, frame_width, frame_height)]
                    ])
                    if self.fixed_bbox_watermark:
                        bbox_line += f" [{self.fixed_bbox_watermark[0]},{self.fixed_bbox_watermark[1]},{self.fixed_bbox_watermark[2]},{self.fixed_bbox_watermark[3]}]"
                else:
                    bbox_line = ""
                txt_file.write(f"Frame_{frame_idx}: {bbox_line}\n")

                for (box, _) in filtered_boxes:
                    dx1, dy1, dx2, dy2 = self.dilate_bounding_box(
                        box, self.dilation_factor, frame_width, frame_height)
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), self.overlay_color, 3)
                
                if self.fixed_bbox_watermark:
                    cv2.rectangle(frame, 
                                (self.fixed_bbox_watermark[0], self.fixed_bbox_watermark[1]), 
                                (self.fixed_bbox_watermark[2], self.fixed_bbox_watermark[3]), 
                                self.overlay_color, 3)
                out.write(frame)
                frame_idx += 1

        cap.release()
        out.release()
        return output_path, txt_output_path
