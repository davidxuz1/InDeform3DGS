import cv2
import os
import argparse

def parse_expression(value):
    """
    Analiza expresiones como '4/60' y las evalúa.
    """
    try:
        return float(eval(value))
    except (SyntaxError, NameError):
        raise argparse.ArgumentTypeError(f"'{value}' no es una expresión válida para un número flotante.")

def main(args):
    # Ruta del archivo de video
    video_path = args.input
    output_path = os.path.join(os.path.dirname(args.input), "video_cut.mp4")
    i = args.i  # 1 para video, 2 para frame

    # Validar argumentos en función de la operación solicitada
    if i == 1 and (args.start is None or args.end is None):
        raise ValueError("Debe proporcionar los argumentos --start y --end para cortar un video.")
    if i == 2 and args.frame is None:
        raise ValueError("Debe proporcionar el argumento --frame para capturar un frame.")

    # Convertir tiempos en minutos si están definidos
    minuto_A = args.start if args.start else 0  # Minuto de inicio (default: 0)
    minuto_B = args.end if args.end else 0     # Minuto final (default: 0)
    minuto_C = args.frame if args.frame else 0 # Minuto del frame (default: 0)

    # Cargar el video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error al abrir el archivo de video.")

    out = None  # Inicializar `out` por seguridad

    try:
        # Propiedades del video
        fps = cap.get(cv2.CAP_PROP_FPS)  # Fotogramas por segundo
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"FPS detectado: {fps}")
        print(f"Duración total del video: {duration:.2f} segundos")
        print(f"Frames totales del video: {total_frames}")

        # Validar tiempos con tolerancia
        tolerancia = 0.1  # Tolerancia para errores de precisión
        if (minuto_A * 60 > duration + tolerancia or
                minuto_B * 60 > duration + tolerancia or
                minuto_C * 60 > duration + tolerancia):
            raise ValueError("Los tiempos definidos exceden la duración del video.")

        # Cálculo de fotogramas usando redondeo
        frame_inicio = round(minuto_A * 60 * fps)
        frame_fin = round(minuto_B * 60 * fps)
        frame_C = round(minuto_C * 60 * fps)

        print(f"Frame de inicio: {frame_inicio}")
        print(f"Frame de fin: {frame_fin}")
        print(f"Frame del minuto C: {frame_C}")

        frames_cortados = frame_fin - frame_inicio + 1
        if i == 1:
            print(f"Frames totales del video cortado: {frames_cortados}")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Saltar directamente al frame de inicio
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio if i == 1 else frame_C)
        frame_count = frame_inicio if i == 1 else frame_C

        while (i == 1 and frame_count <= frame_fin) or (i == 2 and frame_count == frame_C):
            ret, frame = cap.read()

            if not ret:
                print("Se alcanzó el final del video antes de lo esperado.")
                break

            if i == 1:
                out.write(frame)

            if i == 2 and frame_count == frame_C:
                cv2.imwrite(f'frame_minuto_{minuto_C:.2f}.png', frame)
                print(f"Frame del minuto {minuto_C:.2f} guardado como 'frame_minuto_{minuto_C:.2f}.png'.")
                break

            frame_count += 1

    except ValueError as e:
        print(f"Error: {e}")

    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

def parse_arguments():
    # Obtener la ruta del proyecto
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='Seleccionar bounding box de un video o imagen')
    parser.add_argument('--input', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data/input/video.mp4"),
                        help='Ruta al archivo de video o imagen')
    parser.add_argument('--i', type=int, required=True, choices=[1, 2], help='1 para video, 2 para frame')
    parser.add_argument('--start', type=parse_expression, help='Minuto de inicio (ejemplo: 4/60 para 4 segundos)')
    parser.add_argument('--end', type=parse_expression, help='Minuto final (ejemplo: 6/60 para 6 segundos)')
    parser.add_argument('--frame', type=parse_expression, help='Minuto del frame a capturar (ejemplo: 1.5 para 1:30)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
