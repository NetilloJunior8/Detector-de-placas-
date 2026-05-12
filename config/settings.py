"""
config/settings.py
-------------------
Archivo centralizado de configuración del proyecto ALPR.
Todos los parámetros cambiables están aquí. Ningún miembro del equipo
debería tener números mágicos hardcodeados en sus scripts.

Variables de entorno (como la API KEY de Roboflow) van en un archivo '.env'
que está en el .gitignore para no comprometerse en el repositorio.
Las del archivo .env se cargan con python-dotenv.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar .env ANTES de cualquier os.getenv() — orden importa
_PROJECT_ROOT_FOR_ENV = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT_FOR_ENV / ".env")

# ──────────────────────────────────────────────────────────────────────────────
# Rutas del proyecto (todas relativas, sin rutas hardcodeadas de usuario)
# ──────────────────────────────────────────────────────────────────────────────

# Raíz del proyecto = la carpeta que contiene esta carpeta 'config'
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR    = PROJECT_ROOT / "data"
MODELS_DIR  = PROJECT_ROOT / "models"
LOGS_DIR    = PROJECT_ROOT / "logs"
SRC_DIR     = PROJECT_ROOT / "src"

# Subdirectorios de datos
VIDEOS_DIR  = DATA_DIR / "videos"
FRAMES_DIR  = DATA_DIR / "frames"

# Base de datos local de detecciones
DB_PATH     = LOGS_DIR / "detections.db"


# ──────────────────────────────────────────────────────────────────────────────
# Configuración del Modelo de Detección (YOLO)
# ──────────────────────────────────────────────────────────────────────────────

DETECTION_MODEL_NAME = os.getenv("DETECTION_MODEL", "best.pt")
USE_ONNX             = DETECTION_MODEL_NAME.endswith(".onnx")

# Umbral mínimo de confianza para considerar una detección válida (0.0 a 1.0)
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONF", "0.4"))

# Tamaño de imagen de entrada para el modelo (múltiplo de 32).
# Menor = más rápido pero menos preciso. 416 es un buen balance en CPU.
MODEL_IMG_SIZE = int(os.getenv("MODEL_IMG_SIZE", "416"))

# ── Lógica de skip de frames ─────────────────────────────────────────────────
# YOLO se ejecuta 1 de cada N frames para reducir carga en CPU.
# 2 = inferencia en 50% de frames, 3 = 33%, etc.
INFER_EVERY_N = int(os.getenv("INFER_EVERY_N", "2"))


# ──────────────────────────────────────────────────────────────────────────────
# Configuración del OCR (EasyOCR)
# ──────────────────────────────────────────────────────────────────────────────

# Idiomas del OCR. Para Mexico/LATAM el inglés es suficiente para placas de autos.
OCR_LANGUAGES    = ["en"]

# Solo aceptar texto del OCR si la confianza es mayor a este valor (0.0 a 1.0)
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.4"))

# Usar GPU para el OCR (True requiere PyTorch con CUDA. En Matebook, dejarlo en False)
OCR_USE_GPU = False

# Intervalo mínimo entre ejecuciones de OCR (segundos).
# Reducir para mayor frecuencia (más CPU). Aumentar para menos carga.
OCR_INTERVAL_SEC = float(os.getenv("OCR_INTERVAL_SEC", "0.8"))

# Ancho fijo al que se redimensiona el ROI antes de pasar a EasyOCR.
# Menor = más rápido; mayor = más detalle. 320 es un buen balance.
OCR_ROI_WIDTH = int(os.getenv("OCR_ROI_WIDTH", "320"))


# ──────────────────────────────────────────────────────────────────────────────
# Sistema de Votación de Placas (estabilización de resultados OCR)
# ──────────────────────────────────────────────────────────────────────────────

# Número mínimo de veces que debe aparecer un texto OCR para ser confirmado
PLATE_MIN_VOTES = int(os.getenv("PLATE_MIN_VOTES", "3"))

# Tamaño de la ventana de lecturas sobre la que se cuenta la votación
PLATE_VOTE_WINDOW = int(os.getenv("PLATE_VOTE_WINDOW", "7"))

# Longitud mínima del texto de placa para considerarlo válido
PLATE_MIN_LEN = int(os.getenv("PLATE_MIN_LEN", "3"))


# ──────────────────────────────────────────────────────────────────────────────
# Configuración de la Cámara
# ──────────────────────────────────────────────────────────────────────────────

# Índice de la cámara: 0 = cámara integrada, 1 = cámara USB externa
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# Resolución de captura. Reducir si la cámara no puede mantener 30 FPS.
CAMERA_WIDTH  = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
CAMERA_FPS    = int(os.getenv("CAMERA_FPS", "30"))


# ──────────────────────────────────────────────────────────────────────────────
# Configuración del Servidor Flask
# ──────────────────────────────────────────────────────────────────────────────

FLASK_HOST    = "0.0.0.0"  # Permite acceder desde la red local si es necesario
FLASK_PORT    = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG   = False       # NUNCA poner True en producción


# ──────────────────────────────────────────────────────────────────────────────
# Historial en memoria
# ──────────────────────────────────────────────────────────────────────────────

# Cuántas placas recientes guardar en el deque en memoria (para la UI)
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))


ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")  # Definir en .env
