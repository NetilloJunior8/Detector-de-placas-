"""
config/logging_config.py
--------------------------
Sistema de logging centralizado para el proyecto ALPR.
Los logs se guardan simultáneamente en la consola (coloreados)
y en un archivo rotatorio en la carpeta /logs/ del proyecto.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Agregar root del proyecto al path para poder importar settings
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config.settings import LOGS_DIR

# Colores ANSI para consola
RESET   = "\x1b[0m"
GREEN   = "\x1b[32m"
YELLOW  = "\x1b[33m"
RED     = "\x1b[31m"
CYAN    = "\x1b[36m"
BOLD    = "\x1b[1m"

LOG_COLORS = {
    logging.DEBUG:    CYAN,
    logging.INFO:     GREEN,
    logging.WARNING:  YELLOW,
    logging.ERROR:    RED,
    logging.CRITICAL: BOLD + RED,
}


class ColoredFormatter(logging.Formatter):
    """Formatter personalizado que agrega colores ANSI a la consola."""
    def format(self, record):
        color = LOG_COLORS.get(record.levelno, RESET)
        record.levelname = f"{color}{record.levelname:<8}{RESET}"
        return super().format(record)


def setup_logger(name: str = "alpr") -> logging.Logger:
    """
    Configura y retorna un logger listo para usar.
    
    Uso:
        from config.logging_config import setup_logger
        logger = setup_logger(__name__)
        logger.info("Servidor iniciado en el puerto 5000")
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "alpr.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        return logger  # Evitar handlers duplicados al reimportar el módulo

    # --- Handler de Consola (con colores) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = ColoredFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)

    # --- Handler de Archivo (sin colores, máx. 5MB, rotando 3 archivos) ---
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
