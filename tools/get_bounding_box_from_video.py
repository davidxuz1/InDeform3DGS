<<<<<<< HEAD
import os
import sys
import argparse
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Scale
import imageio.v3 as iio
import numpy as np
=======
import cv2
import os
import sys
import argparse

# Obtener la ruta del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class FrameSelector:
    def __init__(self, root, video_path):
        self.root = root
        self.video_path = video_path
        self.frames = iio.imread(video_path, plugin="pyav")
        self.current_frame = 0
        
        # Crear imagen inicial
        self.image = Image.fromarray(self.frames[0])
        self.tk_image = ImageTk.PhotoImage(self.image)
        
        # Canvas para mostrar el frame
        self.canvas = tk.Canvas(root, width=self.image.width, height=self.image.height)
        self.canvas.pack()
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # Slider para seleccionar el frame
        self.slider = Scale(root, from_=0, to=len(self.frames)-1, 
                          orient=tk.HORIZONTAL, command=self.update_frame)
        self.slider.pack(fill=tk.X)
        
        # Botón para confirmar selección
        self.confirm_button = tk.Button(root, text="Seleccionar este frame", 
                                      command=self.confirm_selection)
        self.confirm_button.pack()
        
    def update_frame(self, value):
        self.current_frame = int(value)
        self.image = Image.fromarray(self.frames[self.current_frame])
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.image_on_canvas, image=self.tk_image)
        
    def confirm_selection(self):
        self.root.quit()

class BBoxSelector:
    def __init__(self, root, image):
        self.root = root
        self.image = image
        self.tk_image = ImageTk.PhotoImage(self.image)
        
        self.canvas = tk.Canvas(root, width=self.image.width, height=self.image.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.canvas_width = self.image.width
        self.canvas_height = self.image.height
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, 
            self.start_x, self.start_y, 
            outline='red', width=2
        )
        
    def on_drag(self, event):
        self.end_x, self.end_y = event.x, event.y
        self.canvas.coords(
            self.rect, 
            self.start_x, self.start_y, 
            self.end_x, self.end_y
        )
        
    def on_release(self, event):
        self.end_x, self.end_y = event.x, event.y
        self.root.quit()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Seleccionar bounding box de un video o imagen')
    parser.add_argument('--input', type=str, 
                       default=os.path.join(PROJECT_ROOT, "data/input/video.mp4"),
                       help='Ruta al archivo de video o imagen')
def parse_arguments():
    parser = argparse.ArgumentParser(description='Seleccionar bounding box de un video o imagen')
    parser.add_argument('--input', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data/input/video.mp4"),
                        help='Ruta al archivo de video o imagen')
    return parser.parse_args()

def main():
    args = parse_arguments()
    file_path = args.input

    if not os.path.exists(file_path):
        print(f"Error: El archivo {file_path} no existe.")
        sys.exit(1)

    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == ".mp4":
        # Primero seleccionar el frame
        root_frame = tk.Tk()
        root_frame.title("Seleccionar Frame")
        frame_selector = FrameSelector(root_frame, file_path)
        root_frame.mainloop()
        
        # Obtener la imagen del frame seleccionado
        image = frame_selector.image
        root_frame.destroy()
    elif file_extension in [".jpg", ".jpeg", ".png"]:
        try:
            image = Image.open(file_path)
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
    # Determinar si el archivo es un video o una imagen
    file_extension = os.path.splitext(file_path)[1].lower()

    # Leer el primer cuadro (imagen o primer frame del video)
    frame = None
    if file_extension == ".mp4":
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Error al cargar el video")
            sys.exit(1)
    elif file_extension in [".jpg", ".jpeg", ".png"]:
        frame = cv2.imread(file_path)
        if frame is None:
            print("Error al cargar la imagen")
            sys.exit(1)
    else:
        print("Error: Formato no soportado. Usa un archivo .mp4, .jpg, .jpeg o .png.")
        sys.exit(1)

    # Crear interfaz para seleccionar bbox
    root_bbox = tk.Tk()
    root_bbox.title("Seleccionar Bounding Box")
    selector = BBoxSelector(root_bbox, image)
    root_bbox.mainloop()
    
    # Obtener coordenadas antes de destruir la ventana
    canvas_width = selector.canvas_width
    canvas_height = selector.canvas_height
    root_bbox.destroy()

    # Calcular coordenadas escaladas
    x1 = min(selector.start_x, selector.end_x)
    y1 = min(selector.start_y, selector.end_y)
    x2 = max(selector.start_x, selector.end_x)
    y2 = max(selector.start_y, selector.end_y)
    
    scale_x = image.width / canvas_width
    scale_y = image.height / canvas_height
    
    xmin = int(x1 * scale_x)
    ymin = int(y1 * scale_y)
    xmax = int(x2 * scale_x)
    ymax = int(y2 * scale_y)

    # Seleccionar bounding box manualmente
    bbox = cv2.selectROI("Selecciona el bounding box", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = bbox
    xmin, ymin, xmax, ymax = int(x), int(y), int(x+w), int(y+h)

    # Crear directorio de salida si no existe
    output_dir = os.path.join(PROJECT_ROOT, "data/tools_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar coordenadas
    # Guardar las coordenadas en formato xmin,ymin,xmax,ymax en un archivo .txt
    output_path = os.path.join(output_dir, "fixed_bbox_watermark.txt")
    with open(output_path, "w") as f:
        f.write(f"{xmin} {ymin} {xmax} {ymax}\n")

    # Imprimir el resultado
    print(f"Bounding box guardado en directorio: {output_dir}")
    print(f"Coordenadas: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

if __name__ == "__main__":
    main()




# import os
# import sys
# import argparse
# from PIL import Image, ImageTk
# import tkinter as tk
# from tkinter import filedialog
# import imageio.v3 as iio

# # Obtener la ruta del proyecto
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# class BBoxSelector:
#     def __init__(self, root, image):
#         self.root = root
#         self.image = image
#         self.tk_image = ImageTk.PhotoImage(self.image)
        
#         self.canvas = tk.Canvas(root, width=self.image.width, height=self.image.height)
#         self.canvas.pack()
#         self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
#         self.rect = None
#         self.start_x = None
#         self.start_y = None
#         self.end_x = None
#         self.end_y = None
#         self.canvas_width = self.image.width
#         self.canvas_height = self.image.height
        
#         self.canvas.bind("<ButtonPress-1>", self.on_press)
#         self.canvas.bind("<B1-Motion>", self.on_drag)
#         self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
#     def on_press(self, event):
#         self.start_x = event.x
#         self.start_y = event.y
#         self.rect = self.canvas.create_rectangle(
#             self.start_x, self.start_y, 
#             self.start_x, self.start_y, 
#             outline='red', width=2
#         )
        
#     def on_drag(self, event):
#         self.end_x, self.end_y = event.x, event.y
#         self.canvas.coords(
#             self.rect, 
#             self.start_x, self.start_y, 
#             self.end_x, self.end_y
#         )
        
#     def on_release(self, event):
#         self.end_x, self.end_y = event.x, event.y
#         self.root.quit()

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Seleccionar bounding box de un video o imagen')
#     parser.add_argument('--input', type=str, 
#                        default=os.path.join(PROJECT_ROOT, "data/input/video.mp4"),
#                        help='Ruta al archivo de video o imagen')
#     return parser.parse_args()

# def get_first_frame(video_path):
#     try:
#         frames = iio.imread(video_path, plugin="pyav")
#         return Image.fromarray(frames[0])
#     except Exception as e:
#         print(f"Error al leer el video: {e}")
#         sys.exit(1)

# def main():
#     args = parse_arguments()
#     file_path = args.input

#     if not os.path.exists(file_path):
#         print(f"Error: El archivo {file_path} no existe.")
#         sys.exit(1)

#     # Cargar imagen o primer frame del video
#     file_extension = os.path.splitext(file_path)[1].lower()
    
#     if file_extension == ".mp4":
#         image = get_first_frame(file_path)
#     elif file_extension in [".jpg", ".jpeg", ".png"]:
#         try:
#             image = Image.open(file_path)
#         except Exception as e:
#             print(f"Error al cargar la imagen: {e}")
#             sys.exit(1)
#     else:
#         print("Error: Formato no soportado. Usa un archivo .mp4, .jpg, .jpeg o .png.")
#         sys.exit(1)

#     # Crear interfaz gráfica
#     root = tk.Tk()
#     root.title("Seleccionar Bounding Box")
#     selector = BBoxSelector(root, image)
#     root.mainloop()
    
#     # Obtener coordenadas antes de destruir la ventana
#     canvas_width = selector.canvas_width
#     canvas_height = selector.canvas_height
#     root.destroy()

#     # Calcular coordenadas escaladas
#     x1 = min(selector.start_x, selector.end_x)
#     y1 = min(selector.start_y, selector.end_y)
#     x2 = max(selector.start_x, selector.end_x)
#     y2 = max(selector.start_y, selector.end_y)
    
#     scale_x = image.width / canvas_width
#     scale_y = image.height / canvas_height
    
#     xmin = int(x1 * scale_x)
#     ymin = int(y1 * scale_y)
#     xmax = int(x2 * scale_x)
#     ymax = int(y2 * scale_y)

#     # Crear directorio de salida si no existe
#     output_dir = os.path.join(PROJECT_ROOT, "data/tools_output")
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Guardar coordenadas
#     output_path = os.path.join(output_dir, "fixed_bbox_watermark.txt")
#     with open(output_path, "w") as f:
#         f.write(f"{xmin} {ymin} {xmax} {ymax}\n")

#     print(f"Bounding box guardado en directorio: {output_dir}")
#     print(f"Coordenadas: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

# if __name__ == "__main__":
#     main()















# import cv2
# import os
# import sys
# import argparse

# # Obtener la ruta del proyecto
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Seleccionar bounding box de un video o imagen')
#     parser.add_argument('--input', type=str, 
#                         default=os.path.join(PROJECT_ROOT, "data/input/video.mp4"),
#                         help='Ruta al archivo de video o imagen')
#     return parser.parse_args()

# def main():
#     args = parse_arguments()
#     file_path = args.input

#     if not os.path.exists(file_path):
#         print(f"Error: El archivo {file_path} no existe.")
#         sys.exit(1)

#     # Determinar si el archivo es un video o una imagen
#     file_extension = os.path.splitext(file_path)[1].lower()

#     # Leer el primer cuadro (imagen o primer frame del video)
#     frame = None
#     if file_extension == ".mp4":
#         cap = cv2.VideoCapture(file_path)
#         ret, frame = cap.read()
#         cap.release()
#         if not ret:
#             print("Error al cargar el video")
#             sys.exit(1)
#     elif file_extension in [".jpg", ".jpeg", ".png"]:
#         frame = cv2.imread(file_path)
#         if frame is None:
#             print("Error al cargar la imagen")
#             sys.exit(1)
#     else:
#         print("Error: Formato no soportado. Usa un archivo .mp4, .jpg, .jpeg o .png.")
#         sys.exit(1)

#     # Seleccionar bounding box manualmente
#     bbox = cv2.selectROI("Selecciona el bounding box", frame, fromCenter=False, showCrosshair=True)
#     cv2.destroyAllWindows()

#     x, y, w, h = bbox
#     xmin, ymin, xmax, ymax = int(x), int(y), int(x+w), int(y+h)

#     # Crear directorio de salida si no existe
#     output_dir = os.path.join(PROJECT_ROOT, "data/tools_output")
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Guardar las coordenadas en formato xmin,ymin,xmax,ymax en un archivo .txt
#     output_path = os.path.join(output_dir, "fixed_bbox_watermark.txt")
#     with open(output_path, "w") as f:
#         f.write(f"{xmin} {ymin} {xmax} {ymax}\n")

#     # Imprimir el resultado
#     print(f"Bounding box guardado en directorio: {output_dir}")
#     print(f"Coordenadas: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

# if __name__ == "__main__":
#     main()

