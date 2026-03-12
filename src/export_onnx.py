import sys
from pathlib import Path

# Añadir la raíz del proyecto al path (funciona si se ejecuta desde src/ o desde la raíz)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
import argparse
from config.settings import MODELS_DIR

def export_to_onnx(model_name="yolo11n.pt"):
    """
    Carga un modelo de YOLO (.pt) y lo exporta a formato .onnx para
    mayor velocidad de CPU localmente en las laptops del equipo.
    """
    model_path = MODELS_DIR / model_name
    
    if not model_path.exists():
        print(f"Atención: {model_name} no encontrado en {MODELS_DIR}. Ultralytics lo descargará automáticamente si es la primera vez.")
    
    # Aquí puedes cambiar el path si vas a cargar tu modelo entrenado custom, e.g. "best.pt"
    # model_path puede ser solo el string de nombre o una ruta
    model_to_load = str(model_path) if model_path.exists() else model_name
    
    print(f"Cargando modelo: {model_to_load}")
    model = YOLO(model_to_load)
    
    # Exporta el modelo a formato ONNX. Usa dynamic=True para que acepte imágenes de diversos tamaños si es necesario,
    # pero para YOLO estándar a menudo es mejor un tamaño fijo (ej img_sz=640) para más optimizaciones.
    print("Iniciando exportación a ONNX...")
    path = model.export(format="onnx", imgsz=640, dynamic=False)
    
    print(f"¡Exportación exitosa! El archivo onnx se guardó en: {path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exportar modelo YOLO (.pt) a .onnx')
    parser.add_argument('--model', type=str, default="yolo11n.pt", help='Nombre del modelo a cargar (o pre-entrenado estándar como yolo11n.pt)')
    
    args = parser.parse_args()
    export_to_onnx(args.model)
