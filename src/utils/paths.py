import json
from pathlib import Path

# Raíz del proyecto (3 niveles arriba desde este archivo: src/utils/paths.py -> src/utils -> src -> raíz)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Directorios principales
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Subdirectorios de datos (opcional para mantener orden)
VIDEOS_DIR = DATA_DIR / "videos"
FRAMES_DIR = DATA_DIR / "frames"

def ensure_directories():
    """Asegura que los directorios principales existan (los crea si no)."""
    for directory in [DATA_DIR, MODELS_DIR, SRC_DIR, NOTEBOOKS_DIR, VIDEOS_DIR, FRAMES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print(f"Ruta raíz del proyecto establecida en: {PROJECT_ROOT}")
