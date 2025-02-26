import cv2
import numpy as np
from numba import jit
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time

# Style configuration
BG_COLOR = "#ffffff"
CONTROL_BG = "#9a9cf5"
BUTTON_BG = "#7d7ff2"
TEXT_COLOR = "#2d2d2d"
FONT_NAME = ("Segoe UI", 9) 

class DehazeParams:
    def __init__(self):
        self.omega = 60
        self.t0 = 88
        self.radius = 1
        self.r = 31
        self.alpha = 137
        self.beta = 27

    def get_omega(self): return self.omega / 100.0
    def get_t0(self): return self.t0 / 100.0
    def get_alpha(self): return self.alpha / 100.0

@jit(nopython=True)
def min_channel(src):
    rows, cols, channels = src.shape
    result = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            result[i, j] = min(src[i, j, 0], src[i, j, 1], src[i, j, 2])
    return result

@jit(nopython=True)
def get_min_local_patch(arr, radius):
    rows, cols = arr.shape
    result = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            min_val = arr[i, j]
            for di in range(max(0, i-radius), min(rows, i+radius+1)):
                for dj in range(max(0, j-radius), min(cols, j+radius+1)):
                    if arr[di, dj] < min_val:
                        min_val = arr[di, dj]
            result[i, j] = min_val
    return result

class VideoHazeRemoval:
    def __init__(self, params):
        self.params = params

    @staticmethod
    @jit(nopython=True)
    def _get_dark_channel(src, radius):
        min_rgb = min_channel(src)
        return get_min_local_patch(min_rgb, radius)

    def process_frame(self, frame):
        src = frame.astype(np.float64)/255.
        rows, cols = src.shape[:2]
        
        dark = self._get_dark_channel(src, self.params.radius)
        
        flat_idx = np.argsort(dark.ravel())[-int(dark.size * 0.001):]
        flat_coords = np.unravel_index(flat_idx, dark.shape)
        Alight = src[flat_coords].mean(axis=0)
        Alight = Alight.reshape(1, 1, 3)
        
        tran = np.ones((rows, cols), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                tran[i, j] = 1 - self.params.get_omega() * dark[i, j]
        
        tran = cv2.GaussianBlur(tran.astype(np.float32), (self.params.r, self.params.r), 0)
        tran = np.clip(tran, self.params.get_t0(), 1)
        tran = np.expand_dims(tran, axis=2)
        
        dst = ((src - Alight)/tran + Alight) * 255
        dst = np.clip(dst, 0, 255).astype(np.uint8)
        
        return cv2.convertScaleAbs(dst, alpha=self.params.get_alpha(), beta=self.params.beta)

class DehazeStage:
    def __init__(self):
        self.params = DehazeParams()
        self.input_path = ""
        self.output_path = ""
        
    def process(self, input_video, output_video):
        self.input_path = input_video
        self.output_path = output_video
        
        root = tk.Tk()
        app = DehazeApp(root, self.input_path, self.output_path)
        root.mainloop()
        
        return self.output_path

class DehazeApp:
    def __init__(self, master, input_path, output_path):
        self.master = master
        self.input_path = input_path
        self.output_path = output_path
        self.params = DehazeParams()
        self.processor = VideoHazeRemoval(self.params)
        self.paused = False
        self.current_frame = None
        self.processing_times = []

        self.configure_styles()
        self.setup_gui()
        self.setup_video()
        self.update_frame()

    def configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=FONT_NAME)
        style.configure("TButton", background=BUTTON_BG, foreground=TEXT_COLOR, font=FONT_NAME)
        style.map("TButton", background=[('active', '#6b6df0'), ('pressed', '#5a5cd3')])
        style.configure("TScale", troughcolor=CONTROL_BG, sliderthickness=15)
        style.configure("Video.TLabel", background="#ffffff")

    def setup_gui(self):
        self.master.title("Dehaze Stage - Video Processing")
        self.master.geometry("1280x720")
        self.master.minsize(800, 600)
        self.master.configure(background=BG_COLOR)
        
        # Header
        header = ttk.Frame(self.master)
        header.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(header, text="VIDEO DEHAZER", 
                 font=("Segoe UI", 14, "bold"), 
                 foreground="#4a4cbf").pack()
        
        # Control panel
        control_frame = ttk.Frame(self.master, relief=tk.RIDGE, borderwidth=2)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        

        self.create_slider(control_frame, 'Dehaze Intensity (Ω)', 0, 100, 0)
        self.create_slider(control_frame, 'Detail Preservation (T0)', 0, 100, 1)
        self.create_slider(control_frame, 'Dark Channel Radius', 1, 15, 2)
        self.create_slider(control_frame, 'Kernel Size', 1, 31, 3)
        self.create_slider(control_frame, 'Contrast (α)', 0, 200, 4)
        self.create_slider(control_frame, 'Brightness (β)', 0, 100, 5)

        # Buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=6, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="Pause/Resume", command=self.toggle_pause).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Finish Process", command=self.cleanup).pack(side=tk.LEFT, padx=5)
        
        # Video Panels
        video_frame = ttk.Frame(self.master)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Frame for the Original
        original_container = ttk.Frame(video_frame)
        original_container.grid(row=0, column=0, sticky="nsew", padx=5)
        
        ttk.Label(original_container, text="Original", 
                font=("Segoe UI", 10, "bold")).pack(pady=5)
        self.original_label = ttk.Label(original_container, style="Video.TLabel")
        self.original_label.pack(fill=tk.BOTH, expand=True)

        # Frame for the Processed
        processed_container = ttk.Frame(video_frame)
        processed_container.grid(row=0, column=1, sticky="nsew", padx=5)

        ttk.Label(processed_container, text="Processed", 
                font=("Segoe UI", 10, "bold")).pack(pady=5)
        self.processed_label = ttk.Label(processed_container, style="Video.TLabel")
        self.processed_label.pack(fill=tk.BOTH, expand=True)

        video_frame.grid_columnconfigure(0, weight=1)
        video_frame.grid_columnconfigure(1, weight=1)
        video_frame.grid_rowconfigure(0, weight=1)

    def create_slider(self, frame, text, from_, to, row):
        ttk.Label(frame, text=text, font=FONT_NAME).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        slider = ttk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL,
                          command=lambda v, p=text: self.update_param(v, p))
        slider.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        
        initial_values = {
            'Dehaze Intensity (Ω)': self.params.omega,
            'Detail Preservation (T0)': self.params.t0,
            'Dark Channel Radius': self.params.radius,
            'Kernel Size': self.params.r,
            'Contrast (α)': self.params.alpha,
            'Brightness (β)': self.params.beta
        }
        slider.set(initial_values[text])
    def update_param(self, value, param_name):
        value = int(float(value))
        
        if param_name == 'Kernel Size':
            if value % 2 == 0:
                value += 1
            value = min(value, 31)
            setattr(self.params, 'r', value)
        
        elif param_name == 'Dehaze Intensity (Ω)':
            self.params.omega = value
        
        elif param_name == 'Detail Preservation (T0)':
            self.params.t0 = value
        
        elif param_name == 'Dark Channel Radius':
            self.params.radius = value
        
        elif param_name == 'Contrast (α)':
            self.params.alpha = value
        
        elif param_name == 'Brightness (β)':
            self.params.beta = value
        
        if self.paused and self.current_frame is not None:
            self.process_and_update()

    def setup_video(self):
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise ValueError("Error opening the input video")
                    
        self.writer = cv2.VideoWriter(self.output_path, 
                                    cv2.VideoWriter_fourcc(*'mp4v'), 
                                    self.cap.get(cv2.CAP_PROP_FPS), 
                                    (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    def toggle_pause(self):
        self.paused = not self.paused
        if not self.paused:
            self.process_and_update()

    def process_and_update(self):
        if self.current_frame is not None:
            processed = self.processor.process_frame(self.current_frame)
            self.show_frame(processed, self.processed_label)
            if not self.paused:
                self.writer.write(processed)

    def update_frame(self):
        if not self.paused:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                self.cleanup()
                return
            
            self.current_frame = frame.copy()
            processed = self.processor.process_frame(frame)
            self.writer.write(processed)
            
            self.show_frame(frame, self.original_label)
            self.show_frame(processed, self.processed_label)

            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 10:
                self.processing_times.pop(0)
            
        self.master.after(10, self.update_frame)

    def show_frame(self, frame, label):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        
        original_width, original_height = img_pil.size
        
        if original_width <= 0 or original_height <= 0:
            return
        
        window_width = max(self.master.winfo_width(), 640) 
        window_height = max(self.master.winfo_height(), 480)
        
        max_display_width = max(100, (window_width // 2) - 20)  
        max_display_height = max(100, window_height - 200) 
        
        width_ratio = max_display_width / original_width
        height_ratio = max_display_height / original_height
        scaling_factor = min(width_ratio, height_ratio, 1.0)

        if scaling_factor < 1.0:
            new_size = (
                int(original_width * scaling_factor),
                int(original_height * scaling_factor)
            )
            img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(image=img_pil)
        label.config(image=imgtk)
        label.image = imgtk

    def cleanup(self):
        if self.cap.isOpened():
            self.cap.release()
        if self.writer.isOpened():
            self.writer.release()
        
        if self.processing_times:
            avg_fps = 1 / (sum(self.processing_times) / len(self.processing_times))
            print("\n" + "="*50)
            print("FINAL PROCESS STATISTICS")
            print("="*50)
            print(f"• Average frame processing time: {sum(self.processing_times)/len(self.processing_times)*1000:.2f} ms")
            print(f"• Average FPS: {avg_fps:.2f}")
            
            print("\n" + "="*50)
            print("FINAL PARAMETERS USED")
            print("="*50)
            print(f"• Dehaze Intensity (Ω): {self.params.omega}%")
            print(f"• Detail Preservation (T0): {self.params.t0}%")
            print(f"• Dark Channel Radius: {self.params.radius}")
            print(f"• Kernel Size: {self.params.r} (odd)")
            print(f"• Contrast (α): {self.params.alpha}%")
            print(f"• Brightness (β): {self.params.beta}")
            print("="*50 + "\n")
        
        self.master.destroy()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Dehaze Video Processor')
    parser.add_argument('--input', type=str, default="video.mp4", help='Input video')
    parser.add_argument('--output', type=str, default="dehazed_video.mp4", help='Output video')
    args = parser.parse_args()
    
    DehazeStage().process(args.input, args.output)

if __name__ == "__main__":
    main()
