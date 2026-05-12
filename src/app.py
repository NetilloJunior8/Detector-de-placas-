"""
src/app.py
-----------
Servidor Flask del sistema ALPR.
"""
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from flask import Flask, render_template, Response, jsonify, make_response
from config.settings import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, CAMERA_INDEX
from config.logging_config import setup_logger
from live_detector import ALPRDetector

logger = setup_logger("flask_app")

app = Flask(__name__)

# ─────────────────────────────────────────────────────────
# Inicialización del detector (una sola instancia global)
# ─────────────────────────────────────────────────────────
detector: ALPRDetector | None = None

try:
    logger.info("Inicializando detector ALPR...")
    detector = ALPRDetector()
    detector.start(camera_index=CAMERA_INDEX)
except Exception as e:
    logger.error(f"Error al inicializar el detector: {e}")
    logger.warning("El servidor iniciará sin el detector activo.")


# ─────────────────────────────────────────────────────────
# Rutas principales
# ─────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    if detector is None:
        return "El detector no está disponible.", 503
    return Response(
        detector.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/status')
def status():
    """Endpoint JSON con toda la info en tiempo real para el frontend."""
    if detector is None:
        return jsonify({"plate": "", "history": [], "fps": 0,
                        "conf": 0, "inference_fps": 0, "boxes": []})
    return jsonify({
        "plate":         detector.latest_detected_plate,
        "conf":          round(detector.latest_confidence * 100, 1),
        "history":       detector.detection_history,
        "fps":           round(detector.fps, 1),
        "inference_fps": round(detector.inference_fps, 1),
        "session_id":    detector.session_id,
        "boxes":         detector.latest_boxes,
    })


@app.route('/metrics')
def metrics():
    """Métricas detalladas para el panel de debug."""
    if detector is None:
        return jsonify({"error": "detector no disponible"})
    summary = detector._plate_logger.get_session_summary()
    return jsonify({
        "session_id":    detector.session_id,
        "stream_fps":    round(detector.fps, 1),
        "inference_fps": round(detector.inference_fps, 1),
        "plate":         detector.latest_detected_plate,
        "confidence":    round(detector.latest_confidence * 100, 1),
        "total_detections": summary.get("total", 0),
        "unique_plates":    summary.get("unique", 0),
        "first_seen":       summary.get("first_seen"),
        "last_seen":        summary.get("last_seen"),
    })


@app.route('/export_csv')
def export_csv():
    """Descarga el historial de la sesión actual como archivo CSV."""
    if detector is None:
        return "Detector no disponible.", 503

    csv_data = detector._plate_logger.export_csv()
    filename  = f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    response = make_response(csv_data)
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    return response


# mantener compatibilidad con el endpoint antiguo
@app.route('/latest_plate')
def latest_plate():
    plate = detector.latest_detected_plate if detector else ""
    return jsonify({"plate": plate})


# ─────────────────────────────────────────────────────────
# Punto de Entrada
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info(f"Servidor Flask iniciando en http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        threaded=True,
        use_reloader=False
    )
