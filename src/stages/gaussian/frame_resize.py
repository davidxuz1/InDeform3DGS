import cv2
import os
import shutil
import re

class FrameResizer:
    def __init__(self, input_path, output_folder, target_size=(224, 224), prefix=1):
        self.input_path = input_path
        self.output_folder = output_folder
        self.target_size = target_size
        self.prefix = prefix
        self.original_width = 0
        self.original_height = 0
        self.total_frames = 0
        self.temp_folder = None

    def prepare_output_folder(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def create_temp_folder(self):
        self.temp_folder = os.path.join(os.path.dirname(self.output_folder), "temp_frames")
        if os.path.exists(self.temp_folder):
            shutil.rmtree(self.temp_folder)
        os.makedirs(self.temp_folder)
        return self.temp_folder

    def extract_frames_from_video(self):
        print(f"Extracting frames from video: {self.input_path}")
        temp_folder = self.create_temp_folder()
        cap = cv2.VideoCapture(self.input_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {self.input_path}")
            return None

        self.original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        digit_count = len(str(total_frames))
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_name = self.get_frame_name(frame_count, digit_count)
            frame_path = os.path.join(temp_folder, frame_name)
            cv2.imwrite(frame_path, frame)

            frame_count += 1
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Extracting frames: {progress:.2f}%")

        cap.release()
        self.total_frames = frame_count
        print(f"Extraction completed: {frame_count} frames saved in {temp_folder}")
        return temp_folder

    def get_frame_list(self, folder):
        """Gets the list of image files in the folder, sorted numerically."""
        frames = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        frames.sort(key=lambda x: int(re.search(r'\d+', x).group()))  # Sort by number
        self.total_frames = max(len(frames), 1)
        return frames

    def calculate_padding(self, width, height):
        aspect_ratio = width / height
        target_aspect = self.target_size[0] / self.target_size[1]

        if aspect_ratio > target_aspect:
            new_width = self.target_size[0]
            new_height = int(new_width / aspect_ratio)
            pad_top = (self.target_size[1] - new_height) // 2
            pad_bottom = self.target_size[1] - new_height - pad_top
            pad_left, pad_right = 0, 0
        else:
            new_height = self.target_size[1]
            new_width = int(new_height * aspect_ratio)
            pad_left = (self.target_size[0] - new_width) // 2
            pad_right = self.target_size[0] - new_width - pad_left
            pad_top, pad_bottom = 0, 0

        return new_width, new_height, pad_top, pad_bottom, pad_left, pad_right

    def process_frame(self, frame_name, idx, input_folder, digit_count):
        frame_path = os.path.join(input_folder, frame_name)
        frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)

        if frame is None:
            print(f"Error reading frame: {frame_name}")
            return

        height, width = frame.shape[:2]
        if idx == 0 and self.original_width == 0:
            self.original_width, self.original_height = width, height

        print(f"Processing {frame_name} - Dimensions: {width}x{height}")

        new_width, new_height, pad_top, pad_bottom, pad_left, pad_right = self.calculate_padding(width, height)

        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        padded_frame = cv2.copyMakeBorder(resized_frame, pad_top, pad_bottom, pad_left, pad_right,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])

        final_frame = cv2.resize(padded_frame, self.target_size, interpolation=cv2.INTER_AREA)

        output_name = self.get_frame_name(idx, digit_count)
        output_path = os.path.join(self.output_folder, output_name)
        cv2.imwrite(output_path, final_frame)

        if (idx + 1) % 10 == 0:
            print(f"Progress: {((idx + 1) / self.total_frames) * 100:.2f}%")

    def get_frame_name(self, idx, digit_count):
        extension = ".png"
        if self.prefix == 1:
            return f"frame-{idx:06d}.color{extension}"
        elif self.prefix == 2:
            return f"{idx}{extension}"
        elif self.prefix == 3:
            return f"frame-{idx:06d}.depth{extension}"
        elif self.prefix == 4:
            return f"frame-{idx:06d}.mask{extension}"
        else:
            raise ValueError("The --prefix argument must be 1, 2, 3, or 4.")

    def is_video_file(self, path):
        return os.path.isfile(path) and path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv'))

    def resize_frames(self):
        self.prepare_output_folder()
        input_folder = self.extract_frames_from_video() if self.is_video_file(self.input_path) else self.input_path
        if not input_folder:
            return

        frames = self.get_frame_list(input_folder)
        if not frames:
            print("No images found.")
            return

        print(f"Processing {len(frames)} frames...")
        digit_count = len(str(len(frames)))

        for idx, frame_name in enumerate(frames):
            self.process_frame(frame_name, idx, input_folder, digit_count)

        if self.temp_folder and os.path.exists(self.temp_folder):
            shutil.rmtree(self.temp_folder)

        self.print_summary()

    def print_summary(self):
        print("\nProcessing Summary:")
        print(f"Original resolution: {self.original_width}x{self.original_height}")
        print(f"New resolution: {self.target_size[0]}x{self.target_size[1]}")
        print(f"Frames processed: {self.total_frames}")
        print(f"Frames saved in: {self.output_folder}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video or image frame resizer")
    parser.add_argument("input_path", type=str, help="Path to the video or image folder")
    parser.add_argument("output_folder", type=str, help="Folder where processed frames will be saved")
    parser.add_argument("--size", type=int, nargs=2, default=[224, 224], help="Desired size (width height)")
    parser.add_argument("--prefix", type=int, choices=[1, 2, 3, 4], required=True,
                        help="Naming format: 1=color, 2=simple index, 3=depth, 4=mask")

    args = parser.parse_args()

    resizer = FrameResizer(args.input_path, args.output_folder, tuple(args.size), args.prefix)
    resizer.resize_frames()
