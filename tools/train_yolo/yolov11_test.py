import os
import cv2
import argparse
from ultralytics import YOLO

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Detection Script")
    parser.add_argument("--input", type=str,
                    default=os.path.join(PROJECT_ROOT, "data/input/video.mp4"),
                    help="Path to input video/image (optional)")
    parser.add_argument("--confidence", type=float,
                    default=0.7,
                    help="Confidence threshold for detection")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set input and output paths
    input_path = args.input
    output_dir = os.path.join(PROJECT_ROOT, "data/tools_output/yolo_output")
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLO model
    model = YOLO("best.pt")
    
    # Check if input is image or video
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Processing image from: {input_path}")
        
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not load image from {input_path}")
            return
            
        # Perform detection
        results = model(image)
        
        # Filter detections by confidence
        if results[0].boxes is not None:
            conf_mask = results[0].boxes.conf > args.confidence
            results[0].boxes = results[0].boxes[conf_mask]
            
        # Draw detections
        annotated_image = results[0].plot()
        
        # Save output
        output_image_path = os.path.join(output_dir, "detected_image.png")
        cv2.imwrite(output_image_path, annotated_image)
        print(f"Detection completed. Output saved to: {output_image_path}")
        
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print(f"Processing video from: {input_path}")
        
        # Process video (rest of the video processing code remains the same)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video at {input_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = os.path.join(output_dir, "detected_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            if results[0].boxes is not None:
                conf_mask = results[0].boxes.conf > args.confidence
                results[0].boxes = results[0].boxes[conf_mask]

            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            frame_idx += 1
            print(f"Processed frame {frame_idx}")

        cap.release()
        out.release()
        print(f"Detection completed. Output saved to: {output_video_path}")
        
    else:
        print("Error: Input file must be an image (.png, .jpg, .jpeg) or video (.mp4, .avi, .mov, .mkv)")

if __name__ == "__main__":
    main()
