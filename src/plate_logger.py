"""
src/plate_logger.py
--------------------
Módulo de persistencia de detecciones de placas usando SQLite.

- Logging completamente asíncrono (no bloquea el hilo de inferencia).
- La base de datos se crea automáticamente en LOGS_DIR/detections.db.
- Soporta exportación a CSV y resumen de sesión.
"""
import sys
import csv
import queue
import threading
import sqlite3
import io
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH, LOGS_DIR
from config.logging_config import setup_logger

logger = setup_logger(__name__)


class PlateLogger:
    """
    Logger asíncrono de placas con backend SQLite.

    Uso:
        pl = PlateLogger(session_id="abc123")
        pl.start()
        pl.log("ABC-123", confidence=0.92)
        pl.stop()
        csv_text = pl.export_csv()
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._db_path = DB_PATH

        # Asegurar que el directorio de logs existe
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Inicializar esquema de la base de datos (sincrónico, solo al arrancar)
        self._init_db()

    # ──────────────────────────────────────────────────────────────────────
    # Ciclo de vida
    # ──────────────────────────────────────────────────────────────────────

    def start(self):
        """Inicia el hilo de escritura en background."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="plate-logger"
        )
        self._thread.start()
        logger.info(f"PlateLogger iniciado (session={self.session_id}, db={self._db_path})")

    def stop(self, timeout: float = 3.0):
        """Para el hilo de escritura y drena la cola antes de salir."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("PlateLogger detenido.")

    # ──────────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────────

    def log(self, plate_text: str, confidence: float = 0.0):
        """
        Encola una detección para escritura asíncrona.
        Nunca bloquea el hilo que llama.
        """
        self._queue.put_nowait({
            "plate":      plate_text,
            "confidence": round(confidence, 4),
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "session_id": self.session_id,
        })

    def export_csv(self) -> str:
        """
        Retorna el historial completo de la sesión actual como string CSV.
        Hilo-seguro (usa su propia conexión SQLite).
        """
        rows = self._fetch_session_rows()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "plate", "confidence", "timestamp", "session_id"])
        writer.writerows(rows)
        return output.getvalue()

    def get_session_summary(self) -> dict:
        """Retorna un dict con estadísticas de la sesión actual."""
        rows = self._fetch_session_rows()
        plates = [r[1] for r in rows]
        unique_plates = set(plates)
        return {
            "session_id":    self.session_id,
            "total":         len(rows),
            "unique":        len(unique_plates),
            "plates":        list(unique_plates),
            "first_seen":    rows[0][3] if rows else None,
            "last_seen":     rows[-1][3] if rows else None,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Internos
    # ──────────────────────────────────────────────────────────────────────

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate      TEXT    NOT NULL,
                    confidence REAL    DEFAULT 0.0,
                    timestamp  TEXT    NOT NULL,
                    session_id TEXT    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session ON detections(session_id)"
            )
            conn.commit()

    def _writer_loop(self):
        """Hilo de escritura: drena la cola y escribe en SQLite por lotes."""
        conn = sqlite3.connect(self._db_path)
        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                batch = []
                try:
                    # Esperar hasta 0.5 s por el primer item
                    item = self._queue.get(timeout=0.5)
                    batch.append(item)
                    # Drenar cola sin bloquear (micro-batch)
                    while True:
                        try:
                            batch.append(self._queue.get_nowait())
                        except queue.Empty:
                            break
                except queue.Empty:
                    continue

                if batch:
                    conn.executemany(
                        "INSERT INTO detections (plate, confidence, timestamp, session_id) "
                        "VALUES (:plate, :confidence, :timestamp, :session_id)",
                        batch,
                    )
                    conn.commit()
        finally:
            conn.close()

    def _fetch_session_rows(self) -> list:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT id, plate, confidence, timestamp, session_id "
                "FROM detections WHERE session_id = ? ORDER BY id",
                (self.session_id,),
            )
            return cursor.fetchall()
