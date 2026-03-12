"""
src/live_detector.py
─────────────────────
Detector ALPR de alto rendimiento — 3 hilos + OCR completamente asíncrono.

Arquitectura:
  Thread A : Captura  ──► raw_queue(2)           [siempre a 30 FPS]
  Thread B : YOLO     ──► det_queue(2)            [~5-10 FPS, SIN OCR]
  Thread C : Render   ◄── det_queue(2)            [siempre fluido, JPEG encode]
  OCR Exec : 1 worker ──► actualiza texto vía lock [async, NUNCA bloquea el video]

El video corre siempre a la velocidad de la cámara.
OCR corre en paralelo sin interrumpir el stream.
"""
import re
import sys
import time
import uuid
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import deque, Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

from config.settings import (
    MODELS_DIR, DETECTION_MODEL_NAME, DETECTION_CONFIDENCE,
    OCR_LANGUAGES, OCR_MIN_CONFIDENCE, OCR_USE_GPU,
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    MODEL_IMG_SIZE, INFER_EVERY_N, OCR_INTERVAL_SEC, OCR_ROI_WIDTH,
    PLATE_MIN_VOTES, PLATE_VOTE_WINDOW, PLATE_MIN_LEN, MAX_HISTORY,
)
from config.logging_config import setup_logger
from plate_logger import PlateLogger

logger = setup_logger(__name__)

# Regex para filtrar texto OCR válido (placas México/LATAM)
_PLATE_RE = re.compile(
    r'^[A-Z0-9]{2,4}[-\s]?[A-Z0-9]{2,4}[-\s]?[A-Z0-9]{0,4}$'
)


def _is_valid_plate(text: str) -> bool:
    clean = text.strip().upper().replace(" ", "").replace("-", "")
    if len(clean) < PLATE_MIN_LEN:
        return False
    return bool(_PLATE_RE.match(text.strip().upper()))


def _correct_plate_characters(text: str) -> str:
    """Corrige confusiones comunes del OCR basándose en el contexto del caracter."""
    if not text:
        return text
    chars = list(text.upper().strip())
    for i, c in enumerate(chars):
        if c in ('O', 'Q', 'I', 'Z', 'B', 'S', 'G', 'D'):
            is_num_prev = i > 0 and chars[i-1].isdigit()
            is_num_next = i < len(chars)-1 and chars[i+1].isdigit()
            if is_num_prev or is_num_next:
                if c in ('O', 'Q', 'D'): chars[i] = '0'
                elif c == 'I': chars[i] = '1'
                elif c == 'Z': chars[i] = '2'
                elif c == 'B': chars[i] = '8'
                elif c == 'S': chars[i] = '5'
                elif c == 'G': chars[i] = '6'
        elif c in ('0', '1', '2', '5', '8'):
            is_alpha_prev = i > 0 and chars[i-1].isalpha()
            is_alpha_next = i < len(chars)-1 and chars[i+1].isalpha()
            if is_alpha_prev and is_alpha_next:
                if c == '0': chars[i] = 'O'
                elif c == '1': chars[i] = 'I'
                elif c == '2': chars[i] = 'Z'
                elif c == '5': chars[i] = 'S'
                elif c == '8': chars[i] = 'B'
    return "".join(chars)


class ALPRDetector:
    """
    Detector ALPR con pipeline 3 hilos + OCR executor asíncrono.
    El video nunca se congela independientemente de la velocidad del OCR.
    """

    def __init__(self):
        # ── Modelo YOLO con fallback automático ──────────────────────────────
        model_path = MODELS_DIR / DETECTION_MODEL_NAME
        if model_path.exists():
            model_to_load = str(model_path)
            logger.info(f"Modelo encontrado: {model_path}")
        else:
            fallback_names = ["best.onnx", "best.pt", "yolo11n.pt", "yolov8n.pt"]
            fallback_path = next(
                (MODELS_DIR / n for n in fallback_names if (MODELS_DIR / n).exists()),
                None
            )
            if fallback_path:
                model_to_load = str(fallback_path)
                logger.warning(
                    f"'{DETECTION_MODEL_NAME}' no encontrado. "
                    f"Usando fallback: {fallback_path.name}"
                )
            else:
                model_to_load = DETECTION_MODEL_NAME
                logger.warning(
                    f"Modelo '{DETECTION_MODEL_NAME}' no encontrado en {MODELS_DIR}. "
                    "Intentando descarga automática."
                )

        logger.info(f"Cargando modelo: {model_to_load}")
        self.model = YOLO(model_to_load, task='detect')

        # ── EasyOCR ──────────────────────────────────────────────────────────
        logger.info(f"Cargando EasyOCR (idiomas: {OCR_LANGUAGES}, GPU: {OCR_USE_GPU})")
        self.reader = easyocr.Reader(OCR_LANGUAGES, gpu=OCR_USE_GPU)

        # ── CLAHE para pre-procesamiento OCR ─────────────────────────────────
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        # ── Colas inter-hilo ──────────────────────────────────────────────────
        self._raw_queue: queue.Queue = queue.Queue(maxsize=2)
        self._det_queue: queue.Queue = queue.Queue(maxsize=2)

        # ── Estado compartido (lock mínimo) ───────────────────────────────────
        self._lock = threading.Lock()
        self._latest_frame: bytes | None = None
        self._latest_plate: str          = ""
        self._latest_conf: float         = 0.0
        self._detection_history: deque   = deque(maxlen=MAX_HISTORY)
        self._stream_fps: float          = 0.0
        self._infer_fps: float           = 0.0
        self._running: bool              = False

        # ── OCR asíncrono via executor ────────────────────────────────────────
        # max_workers=1 garantiza que OCR nunca corre en paralelo consigo mismo
        self._ocr_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="alpr-ocr"
        )
        self._ocr_future = None   # solo accedido desde _detection_thread

        # ── Votación (solo desde OCR executor, con max_workers=1 es thread-safe) ─
        self._vote_window: deque = deque(maxlen=PLATE_VOTE_WINDOW)

        # ── Logger SQLite asíncrono ───────────────────────────────────────────
        self._session_id = uuid.uuid4().hex[:8]
        self._plate_logger = PlateLogger(session_id=self._session_id)

        logger.info("ALPRDetector listo.")

    # ─── Propiedades públicas (thread-safe) ──────────────────────────────────

    @property
    def latest_detected_plate(self) -> str:
        with self._lock:
            return self._latest_plate

    @property
    def latest_confidence(self) -> float:
        with self._lock:
            return self._latest_conf

    @property
    def detection_history(self) -> list:
        with self._lock:
            return list(reversed(self._detection_history))

    @property
    def fps(self) -> float:
        with self._lock:
            return self._stream_fps

    @property
    def inference_fps(self) -> float:
        with self._lock:
            return self._infer_fps

    @property
    def session_id(self) -> str:
        return self._session_id

    # ─── Pre-procesamiento y OCR (corre en executor thread) ──────────────────

    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """CLAHE + Binarización Adaptativa + morphological opening."""
        if roi.size == 0:
            return roi
        h, w = roi.shape[:2]
        if w > 0:
            scale = OCR_ROI_WIDTH / w
            new_h = max(1, int(h * scale))
            roi = cv2.resize(roi, (OCR_ROI_WIDTH, new_h), interpolation=cv2.INTER_LANCZOS4)
        gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)
        blur = cv2.bilateralFilter(enhanced, 11, 17, 17)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
        )
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    def _ocr_box(self, frame: np.ndarray, x1: int, y1: int,
                 x2: int, y2: int) -> str:
        h, w = frame.shape[:2]
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = int(box_w * 0.15)
        pad_y = int(box_h * 0.25)
        
        roi  = frame[max(0, y1 - pad_y): min(h, y2 + pad_y),
                     max(0, x1 - pad_x): min(w, x2 + pad_x)]
        proc = self._preprocess_roi(roi)
        if proc.size == 0:
            return ""
        results = self.reader.readtext(proc, detail=1, paragraph=False)
        texts   = [r[1].upper().strip() for r in results if r[2] >= OCR_MIN_CONFIDENCE]
        raw_text = " ".join(texts).strip()
        return _correct_plate_characters(raw_text)

    def _vote_plate(self, candidate: str) -> str | None:
        """Solo llamar desde OCR executor (max_workers=1 → thread-safe)."""
        self._vote_window.append(candidate)
        counts = Counter(self._vote_window)
        best, best_count = counts.most_common(1)[0]
        if best_count >= PLATE_MIN_VOTES and _is_valid_plate(best):
            return best
        return None

    def _run_ocr_job(self, frame: np.ndarray, boxes: list) -> list:
        """
        Corre en el OCR executor thread.
        Retorna lista actualizada de (x1,y1,x2,y2,text,conf,track_id).
        Actualiza el estado de placa detectada vía self._lock.
        """
        updated = []
        for box in boxes:
            x1, y1, x2, y2, _, conf = box[:6]
            track_id = box[6] if len(box) > 6 else -1
            
            raw_text  = self._ocr_box(frame, x1, y1, x2, y2)
            confirmed = self._vote_plate(raw_text) if raw_text else None
            text      = confirmed if confirmed else raw_text
            updated.append((x1, y1, x2, y2, text, conf, track_id))

            if confirmed:
                with self._lock:
                    if self._latest_plate != confirmed:
                        self._latest_plate = confirmed
                        self._latest_conf  = conf
                        if (not self._detection_history
                                or self._detection_history[-1] != confirmed):
                            self._detection_history.append(confirmed)
                            logger.info(
                                f"✔ Placa confirmada: {confirmed} "
                                f"(conf={conf:.2f})"
                            )
                self._plate_logger.log(confirmed, confidence=conf)
        return updated

    # ─── HILO A: Captura ─────────────────────────────────────────────────────

    def _capture_thread(self, camera_index: int):
        """Lee frames de la cámara y los encola. Simple y rápido."""
        cap = None

        # Intentar backends disponibles en Windows
        for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            try:
                c = cv2.VideoCapture(camera_index, api)
                if c.isOpened():
                    cap = c
                    logger.info(f"Cámara {camera_index} abierta (api={api})")
                    break
                c.release()
            except Exception:
                pass

        if cap is None or not cap.isOpened():
            logger.error(
                f"No se pudo abrir la cámara {camera_index}. "
                "Verifica CAMERA_INDEX en .env (prueba 0 o 1) "
                "y que ninguna otra app la esté usando."
            )
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Resolución activa: {actual_w}×{actual_h}")

        fps_frames = 0
        fps_timer  = time.time()
        err_count  = 0

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    err_count += 1
                    if err_count % 30 == 1:
                        logger.warning(f"cap.read() falló ({err_count} veces). "
                                       "Reintentando...")
                    time.sleep(0.01)
                    continue

                err_count  = 0
                fps_frames += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    with self._lock:
                        self._stream_fps = fps_frames / (now - fps_timer)
                    fps_frames = 0
                    fps_timer  = now

                # Descartar el frame más viejo si la cola está llena
                if self._raw_queue.full():
                    try:
                        self._raw_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self._raw_queue.put_nowait(frame)
                except queue.Full:
                    pass
        finally:
            cap.release()
            logger.info("Hilo de captura terminado.")

    # ─── HILO B: Detección YOLO (SIN OCR, siempre veloz) ─────────────────────

    def _detection_thread(self):
        """
        Ejecuta YOLO cada INFER_EVERY_N frames.
        Lanza OCR de forma asíncrona (no espera su resultado para enviar al render).
        El video SIEMPRE fluye a la velocidad de este hilo (~10-15 FPS en CPU).
        """
        logger.info("Hilo de detección iniciado.")

        frame_count   = 0
        last_ocr_time = 0.0
        cached_boxes: list = []   # [(x1,y1,x2,y2,text,conf,track_id)]
        infer_frames  = 0
        infer_timer   = time.time()

        try:
            while self._running:
                try:
                    frame = self._raw_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                frame_count += 1
                now = time.time()

                # ── YOLO cada INFER_EVERY_N frames ───────────────────────────
                if frame_count % INFER_EVERY_N == 0:
                    yolo_results = self.model.track(
                        frame,
                        imgsz=MODEL_IMG_SIZE,
                        stream=True,
                        conf=DETECTION_CONFIDENCE,
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False,
                    )
                    new_boxes = []
                    for result in yolo_results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            track_id = int(box.id[0]) if box.id is not None else -1
                            
                            # Conservar texto OCR del mismo track_id (prioridad), si no existe o -1, por distancia
                            prev_text = ""
                            if track_id != -1 and cached_boxes:
                                prev_text = next((b[4] for b in cached_boxes if len(b) > 6 and b[6] == track_id), "")
                            if not prev_text and cached_boxes:
                                prev_text = next(
                                    (b[4] for b in cached_boxes
                                     if abs(b[0] - x1) < 60 and abs(b[1] - y1) < 60),
                                    ""
                                )
                            new_boxes.append((x1, y1, x2, y2, prev_text, conf, track_id))
                    cached_boxes = new_boxes

                    infer_frames += 1
                    elapsed = now - infer_timer
                    if elapsed >= 1.0:
                        with self._lock:
                            self._infer_fps = infer_frames / elapsed
                        infer_frames = 0
                        infer_timer  = now

                    # ── Lanzar OCR async si hay placas y se cumplió el intervalo ─
                    if (cached_boxes
                            and (now - last_ocr_time) >= OCR_INTERVAL_SEC
                            and (self._ocr_future is None
                                 or self._ocr_future.done())):
                        last_ocr_time  = now
                        self._ocr_future = self._ocr_executor.submit(
                            self._run_ocr_job,
                            frame.copy(),
                            list(cached_boxes),
                        )

                # ── Recoger resultado OCR si ya terminó (no-blocking) ─────────
                if self._ocr_future is not None and self._ocr_future.done():
                    try:
                        ocr_updated = self._ocr_future.result(timeout=0)
                        # Actualizar texto preservando posiciones YOLO actuales
                        text_map_track = {b[6]: b[4] for b in ocr_updated if len(b) > 6 and b[6] != -1}
                        text_map_pos = {(b[0], b[1]): b[4] for b in ocr_updated}
                        
                        new_cached = []
                        for b in cached_boxes:
                            x1, y1, x2, y2, text, conf = b[:6]
                            track_id = b[6] if len(b) > 6 else -1
                            new_text = text
                            if track_id != -1 and track_id in text_map_track:
                                new_text = text_map_track[track_id]
                            elif (x1, y1) in text_map_pos:
                                new_text = text_map_pos[(x1, y1)]
                            new_cached.append((x1, y1, x2, y2, new_text, conf, track_id))
                        cached_boxes = new_cached
                    except Exception as exc:
                        logger.warning(f"OCR job error: {exc}")
                    self._ocr_future = None

                # ── Enviar frame + cajas al render (sin esperar OCR) ──────────
                snap = (frame.copy(), list(cached_boxes))
                if self._det_queue.full():
                    try:
                        self._det_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self._det_queue.put_nowait(snap)
                except queue.Full:
                    pass

        finally:
            logger.info("Hilo de detección terminado.")

    # ─── HILO C: Render + JPEG encode ────────────────────────────────────────

    def _render_thread(self):
        """
        Toma (frame, boxes) del det_queue, dibuja overlays, codifica JPEG.
        Nunca toca el modelo ni el OCR → siempre rápido.
        """
        logger.info("Hilo de render iniciado.")
        try:
            while self._running:
                try:
                    frame, boxes = self._det_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                for box in boxes:
                    x1, y1, x2, y2, text, conf = box[:6]
                    # Color dinámico según confianza
                    green = int(220 * conf)
                    red   = int(220 * (1 - conf))
                    color = (0, green, red)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Badge de confianza
                    conf_label = f"{conf:.0%}"
                    (lw, lh), _ = cv2.getTextSize(
                        conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                    )
                    cv2.rectangle(
                        frame, (x1, y1 - lh - 10), (x1 + lw + 8, y1), color, -1
                    )
                    cv2.putText(
                        frame, conf_label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
                    )

                    if text:
                        (tw, th), _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
                        )
                        overlay = frame.copy()
                        cv2.rectangle(
                            overlay,
                            (x1, y2 + 2), (x1 + tw + 8, y2 + th + 12),
                            (0, 0, 0), -1
                        )
                        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                        cv2.putText(
                            frame, text, (x1 + 4, y2 + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
                        )

                # FPS overlay
                with self._lock:
                    fps_val   = self._stream_fps
                    infer_val = self._infer_fps

                cv2.putText(
                    frame,
                    f"CAM {fps_val:.0f} | INF {infer_val:.0f} FPS",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2
                )

                ret, buffer = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if ret:
                    with self._lock:
                        self._latest_frame = buffer.tobytes()

        finally:
            logger.info("Hilo de render terminado.")

    # ─── API pública para Flask ───────────────────────────────────────────────

    def start(self, camera_index: int = CAMERA_INDEX):
        """Inicia los 3 hilos + el OCR executor en background."""
        self._running = True
        self._plate_logger.start()

        for target, name in [
            (lambda: self._capture_thread(camera_index), "alpr-capture"),
            (self._detection_thread,                     "alpr-detection"),
            (self._render_thread,                        "alpr-render"),
        ]:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()

        logger.info("3 hilos ALPR + OCR executor iniciados.")

    def stop(self):
        """Detiene todos los hilos y el executor de OCR."""
        self._running = False
        self._ocr_executor.shutdown(wait=False, cancel_futures=True)
        self._plate_logger.stop()

    def generate_frames(self):
        """
        Generador MJPEG para Flask.
        Solo lee latest_frame (bytes) — nunca bloquea esperando inferencia.
        """
        while True:
            with self._lock:
                frame_bytes = self._latest_frame

            if frame_bytes is None:
                time.sleep(0.01)
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + frame_bytes
                + b'\r\n'
            )
