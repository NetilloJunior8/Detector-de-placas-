import sys
import cv2
import argparse
from pathlib import Path

# Añadir la raíz del proyecto al path (funciona si se ejecuta desde src/ o desde la raíz)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import VIDEOS_DIR, FRAMES_DIR


def ensure_directories():
    """Asegura que los directorios de datos existan."""
    for d in [VIDEOS_DIR, FRAMES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def extract_frames(video_name, frame_rate_divisor=10):
    """
    Extrae frames de un video para el dataset.
    
    Args:
        video_name (str): Nombre del archivo de video (ej. 'autos.mp4') en /data/videos/
        frame_rate_divisor (int): Extrae 1 frame de cada N. 
            Si tu video va a 30fps y extraes cada 10, obtienes 3 frames por segundo.
            Útil para no generar miles de imágenes iguales.
    """
    ensure_directories()
    
    video_path = VIDEOS_DIR / video_name
    
    if not video_path.exists():
        print(f"Error: No se encontró el video {video_path}")
        return
        
    output_dir = FRAMES_DIR / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    saved_count = 0
    
    print(f"Procesando {video_name}...")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        if count % frame_rate_divisor == 0:
            frame_filename = output_dir / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(frame_filename), image)
            saved_count += 1
            
        count += 1

    cap.release()
    print(f"Finalizado. Se extrajeron {saved_count} frames y se guardaron en {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extraer frames de un video.')
    parser.add_argument('video_name', type=str, help='Nombre del video dentro de la carpeta data/videos/')
    parser.add_argument('--divisor', type=int, default=10, help='Extraer 1 de cada N frames. Por defecto: 10')
    
    args = parser.parse_args()
    extract_frames(args.video_name, args.divisor)
