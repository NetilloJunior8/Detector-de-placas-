"""
src/live_detector.py
─────────────────────
Detector ALPR de alto rendimiento — 3 hilos + OCR completamente asíncrono.

Arquitectura:
  Thread A : Captura  ──► raw_queue(2)           [siempre a 30 FPS]
  Thread B : YOLO     ──► det_queue(2)            [~10-15 FPS, SIN OCR]
  Thread C : Render   ◄── det_queue(2)            [siempre fluido, JPEG encode]
  OCR Exec : 1 worker ──► actualiza texto vía lock [async, NUNCA bloquea el video]

OCR Engine: PaddleOCR (PP-OCRv4 mobile, CPU) — 5-10x más rápido que EasyOCR.
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
from paddleocr import PaddleOCR
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

# Regex universal para placas: permite combinaciones alfanuméricas de 4 a 9 caracteres
_PLATE_RE = re.compile(r'^[A-Z0-9]{4,9}$')


def _is_valid_plate(text: str) -> bool:
    # Normalizar: quitar espacios y guiones antes de validar
    clean = text.strip().upper().replace(" ", "").replace("-", "")
    if len(clean) < PLATE_MIN_LEN:
        return False
        
    # Aplicar el regex sobre el texto LIMPIO
    return bool(_PLATE_RE.match(clean))


def _correct_plate_characters(text: str) -> str:
    """Corrige confusiones comunes del OCR basándose en el contexto del caracter."""
    if not text:
        return text
    chars = list(text.upper().strip())
    for i, c in enumerate(chars):
        # Determinar si el caracter está en los bordes de la placa
        is_edge = (i == 0) or (i == len(chars) - 1)
        
        is_num_prev = i > 0 and chars[i-1].isdigit()
        is_num_next = i < len(chars)-1 and chars[i+1].isdigit()
        
        is_alpha_prev = i > 0 and chars[i-1].isalpha()
        is_alpha_next = i < len(chars)-1 and chars[i+1].isalpha()

        if c in ('O', 'Q', 'I', 'Z', 'B', 'S', 'G', 'D'):
            # Convertir letra a número
            should_change_to_num = False
            if is_edge:
                should_change_to_num = is_num_prev or is_num_next
            else:
                # Si está en medio, es más seguro cambiar solo si AMBOS lados son números
                # o si la mayoría de su contexto lo sugiere.
                should_change_to_num = is_num_prev and is_num_next
                
            if should_change_to_num:
                if c in ('O', 'Q', 'D'): chars[i] = '0'
                elif c == 'I': chars[i] = '1'
                elif c == 'Z': chars[i] = '2'
                elif c == 'B': chars[i] = '8'
                elif c == 'S': chars[i] = '5'
                elif c == 'G': chars[i] = '6'

        elif c in ('0', '1', '2', '5', '8'):
            # Convertir número a letra
            should_change_to_alpha = False
            if is_edge:
                should_change_to_alpha = is_alpha_prev or is_alpha_next
            else:
                should_change_to_alpha = is_alpha_prev and is_alpha_next

            if should_change_to_alpha:
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

        # ── PaddleOCR ────────────────────────────────────────────────────────────
        logger.info(f"Cargando PaddleOCR (GPU: {OCR_USE_GPU})")
        self.reader = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=OCR_USE_GPU, show_log=False)

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
        self._latest_boxes: list         = []   # BUG FIX: inicializar para evitar AttributeError
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
    def latest_boxes(self) -> list:
        with self._lock:
            return self._latest_boxes

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
        """
        Preprocesamiento OCR robusto para placas en condiciones adversas:
        1. Resize con Lanczos para mantener nitidez
        2. Escala de grises
        3. CLAHE primero — normaliza iluminación ANTES de sharpen (fix bug orden)
        4. Denoise leve
        5. Sharpen — ahora sobre imagen ya normalizada
        6. Threshold adaptativo — maneja iluminación no uniforme (sombras, placa chueca)
        Retorna tanto la versión binarizada como la imagen CLAHE (intentará ambas con EasyOCR).
        """
        if roi.size == 0:
            return roi
        h, w = roi.shape[:2]
        if w < 10 or h < 5:   # ROI demasiado pequeño — skip
            return roi
        # 1. Resize manteniendo proporción
        scale = OCR_ROI_WIDTH / w
        new_h = max(1, int(h * scale))
        roi = cv2.resize(roi, (OCR_ROI_WIDTH, new_h), interpolation=cv2.INTER_LANCZOS4)
        # 2. Escala de grises
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 3. CLAHE PRIMERO — normalizar iluminación desigual antes de cualquier filtro
        enhanced = self._clahe.apply(gray)
        # 4. Denoise leve para reducir ruido de cámara
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        # 5. Sharpen sobre imagen ya normalizada
        _kernel_sharp = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]], dtype=np.float32)
        sharpened = cv2.filter2D(enhanced, -1, _kernel_sharp)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        # 6. Threshold ADAPTATIVO — robusto a iluminación no uniforme y placas torcidas
        # Incrementamos blockSize para evitar romper trazos gruesos (21 en lugar de 15)
        binary = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=21, C=8
        )
        # Retornamos tanto la versión en escala de grises realzada (excelente para redes neuronales como EasyOCR)
        # como la versión binarizada (buena para alto contraste)
        return enhanced, binary

    def _ocr_box(self, roi_crop: np.ndarray) -> str:
        """
        Recibe el ROI ya recortado. Intenta OCR sobre la imagen preprocesada con PaddleOCR.
        """
        if roi_crop is None or roi_crop.size == 0:
            return ""

        enhanced, binary = self._preprocess_roi(roi_crop)

        def _process_results(results_list, min_conf):
            if not results_list or not results_list[0]:
                return ""
            
            formatted_results = []
            for line in results_list[0]:
                if line:
                    box, (text, conf) = line
                    formatted_results.append((box, text, conf))
                    
            valid = [r for r in formatted_results if r[2] >= min_conf]
            if not valid:
                return ""
            heights = [max(pt[1] for pt in r[0]) - min(pt[1] for pt in r[0]) for r in valid]
            max_h = max(heights)
            texts = [r[1].upper().strip() for r, h in zip(valid, heights) if h >= max_h * 0.55]
            # Limpiar texto (solo alfanumérico) para evitar guiones de tornillos
            clean_texts = ["".join(c for c in t if c.isalnum()) for t in texts]
            return "".join(clean_texts).strip()

        best_text = ""
        try:
            results = self.reader.ocr(enhanced, cls=False)
            best_text = _process_results(results, OCR_MIN_CONFIDENCE)
        except Exception as e:
            logger.debug(f"OCR intento 1 falló: {e}")

        # Intento 2 (fallback): Imagen binarizada
        if not best_text:
            try:
                results2 = self.reader.ocr(binary, cls=False)
                best_text = _process_results(results2, max(0.15, OCR_MIN_CONFIDENCE - 0.15))
            except Exception as e:
                logger.debug(f"OCR intento 2 falló: {e}")

        result = _correct_plate_characters(best_text)
        if result:
            logger.debug(f"OCR raw result: '{result}'")
        return result

    def _vote_plate(self, candidate: str) -> str | None:
        """Solo llamar desde OCR executor (max_workers=1 → thread-safe).
        Normaliza el candidato antes de votar para evitar que 'ABC-123' y 'ABC123'
        cuenten como lecturas distintas.
        """
        # Normalizar: quitar guiones y espacios antes de acumular en la ventana
        normalized = candidate.strip().upper().replace("-", "").replace(" ", "")
        self._vote_window.append(normalized)
        counts = Counter(self._vote_window)
        best, best_count = counts.most_common(1)[0]
        if best_count >= PLATE_MIN_VOTES and _is_valid_plate(best):
            return best
        return None

    def _run_ocr_job(self, roi_crops: list, boxes: list) -> list:
        """
        Corre en el OCR executor thread.
        Recibe roi_crops (recortes de imagen ya extraídos) en lugar del frame completo
        para reducir el uso de memoria y evitar copiar frames grandes.
        Retorna lista actualizada de (x1,y1,x2,y2,text,conf,track_id).
        Actualiza el estado de placa detectada vía self._lock.
        """
        updated = []
        for roi_crop, box in zip(roi_crops, boxes):
            x1, y1, x2, y2, prev_text, conf = box[:6]
            track_id = box[6] if len(box) > 6 else -1

            raw_text  = self._ocr_box(roi_crop)
            confirmed = self._vote_plate(raw_text) if raw_text else None
            
            # Lógica Anti-Parpadeo (Flicker Prevention):
            # Si ya tenemos una placa confirmada de los últimos frames, la mantenemos.
            # Si no, si obtuvimos texto raw que parece placa, lo mostramos.
            # Si obtenemos basura OCR ("letras nada que ver"), preferimos mantener el texto anterior.
            text = prev_text
            if confirmed:
                text = confirmed
            elif raw_text:
                if _is_valid_plate(raw_text):
                    text = raw_text
                elif not prev_text:
                    text = raw_text
            
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
                    # BUG FIX: usar predict() con stream=False en lugar de track(stream=True)
                    # track+stream=True puede retornar generator vacío en single-frame inference.
                    # predict() es más estable y retorna resultados directamente.
                    try:
                        yolo_results = self.model.predict(
                            frame,
                            imgsz=MODEL_IMG_SIZE,
                            conf=DETECTION_CONFIDENCE,
                            verbose=False,
                        )
                    except Exception as yolo_err:
                        logger.warning(f"YOLO predict error: {yolo_err}")
                        yolo_results = []

                    new_boxes = []
                    total_raw = 0
                    for result in yolo_results:
                        if result.boxes is None:
                            continue
                        total_raw += len(result.boxes)
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            track_id = -1  # predict() no tiene IDs de track; se usa posición

                            # Conservar texto OCR previo por proximidad de posición
                            prev_text = ""
                            if cached_boxes:
                                prev_text = next(
                                    (b[4] for b in cached_boxes
                                     if abs(b[0] - x1) < 80 and abs(b[1] - y1) < 80),
                                    ""
                                )
                            new_boxes.append((x1, y1, x2, y2, prev_text, conf, track_id))

                    # Log de diagnóstico — visible en consola para verificar que YOLO funciona
                    if frame_count % (INFER_EVERY_N * 15) == 0:  # cada ~15 inferencias
                        logger.info(
                            f"[YOLO] frame={frame_count} | raw_boxes={total_raw} "
                            f"| cached={len(new_boxes)} | conf_thresh={DETECTION_CONFIDENCE}"
                        )

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
                        last_ocr_time = now
                        # Extraer solo los ROI crops (no copiar el frame completo)
                        # Esto reduce el uso de memoria: solo las regiones de placa pasan al executor
                        fh, fw = frame.shape[:2]
                        roi_crops = []
                        for b in cached_boxes:
                            bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
                            bw  = max(1, bx2 - bx1)
                            bh  = max(1, by2 - by1)
                            # Añadimos un padding pequeño (+5%)
                            # Esto da margen a las letras para que PaddleOCR las reconozca,
                            # sin ser excesivo como antes (+25%) que metía los marcos.
                            pad_x = int(bw * 0.05)
                            pad_y = int(bh * 0.05)
                            
                            # Asegurarnos de que el recorte no se voltee o sea inválido
                            start_y = max(0, by1 - pad_y)
                            end_y = min(fh, by2 + pad_y)
                            start_x = max(0, bx1 - pad_x)
                            end_x = min(fw, bx2 + pad_x)
                            
                            if end_y > start_y and end_x > start_x:
                                crop = frame[start_y:end_y, start_x:end_x].copy()
                                roi_crops.append(crop)
                            else:
                                roi_crops.append(frame[max(0, by1):min(fh, by2), max(0, bx1):min(fw, bx2)].copy())
                        self._ocr_future = self._ocr_executor.submit(
                            self._run_ocr_job,
                            roi_crops,
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
        Codifica a JPEG crudo sin dibujar NADA encima.
        Actualiza latest_boxes para que el cliente (Frontend) dibuje los rectángulos.
        """
        logger.info("Hilo de render iniciado (Modo Zero-Cost CLI/Frontend).")
        try:
            while self._running:
                try:
                    frame, boxes = self._det_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Cero operaciones de dibujo (cv2.rectangle, cv2.putText).
                # Solo codificamos a JPG para mandar por red lo más rápido posible.
                ret, buffer = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if ret:
                    with self._lock:
                        self._latest_frame = buffer.tobytes()
                        # Extraer info para el frontend (JSON serializable)
                        self._latest_boxes = [
                            {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], 
                             "text": b[4], "conf": b[5], "track_id": b[6] if len(b) > 6 else -1}
                            for b in boxes
                        ]
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
