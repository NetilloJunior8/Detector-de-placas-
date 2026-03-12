# 🚗 ALPR Vision — Reconocimiento Automático de Matrículas

Sistema de detección y lectura de placas vehiculares en tiempo real usando **YOLO11n** y **EasyOCR**, servido a través de un dashboard web con **Flask**.

Diseñado para trabajo colaborativo entre 3 personas con **zero conflictos de entorno**.

---

## 🏗️ Arquitectura del Sistema

```
[Cámara Webcam]
    │
    ▼
[YOLO11n Nano]         ← Modelo de Detección (src/live_detector.py)
    │ Bounding Box de la Placa
    ▼
[EasyOCR]              ← Motor de Lectura de Texto
    │ String de la Placa
    ▼
[Flask Server]         ← Backend Web (src/app.py)
    │ Stream MJPEG + API REST
    ▼
[Dashboard Web]        ← Frontend (src/templates/index.html)
```

## 🗂️ Estructura del Proyecto

```
ProyectoPlacasV1/
│
├── config/                    # Configuración centralizada del proyecto
│   ├── settings.py            ← ★ AQUÍ van todos los parámetros (modelo, puertos, confianza)
│   └── logging_config.py      ← Sistema de logs (consola coloreada + archivo rotatorio)
│
├── src/                       # Código fuente Python
│   ├── app.py                 ← Punto de entrada del servidor Flask
│   ├── live_detector.py       ← Clase ALPRDetector (YOLO + EasyOCR → stream JPEG)
│   ├── extract_frames.py      ← Utilidad para extraer frames de videos para el dataset
│   ├── export_onnx.py         ← Convierte .pt a .onnx (ejecutar post-entrenamiento)
│   ├── templates/             
│   │   └── index.html         ← Dashboard Web principal
│   └── static/                
│       └── style.css          ← Estilos del dashboard (dark mode profesional)
│
├── notebooks/
│   └── Train_YOLO11n_ALPR.ipynb  ← ★ Notebook para entrenar en Google Colab con GPU
│
├── models/                    # Pesos del modelo (NO se suben a Git, ver .gitignore)
│   └── .gitkeep
│
├── data/                      # Datos locales (videos/frames — tampoco en Git)
│   └── .gitkeep
│
├── logs/                      # Logs generados en ejecución
│   └── .gitkeep
│
├── .env.example               # ★ Plantilla de variables de entorno (copiar a .env)
├── .gitignore                 # Excluye modelos, entornos virtuales, .env, etc.
├── requirements.txt           # Dependencias del proyecto
├── CONTRIBUTING.md            # Normas de Git para el equipo
└── .github/
    └── PULL_REQUEST_TEMPLATE.md
```

## 🚀 Inicio Rápido

### 1. Clonar y configurar el entorno
```bash
git clone <url-del-repositorio>
cd ProyectoPlacasV1

python -m venv venv
.\venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Configurar variables de entorno
```bash
copy .env.example .env       # Windows
# Luego abre .env y edita los valores que necesites
```

### 3. Ejecutar el servidor
```bash
python src/app.py
```
Abrir en el navegador: **[http://localhost:5000](http://localhost:5000)**

---

## 🧠 Flujo de Entrenamiento (Google Colab)

El modelo **no se entrena localmente** para no saturar las laptops del equipo.

1. Sube `notebooks/Train_YOLO11n_ALPR.ipynb` a **Google Colab**.
2. Selecciona entorno de ejecución **T4 GPU** (gratis).
3. Descarga tu dataset de [Roboflow Universe](https://universe.roboflow.com/) buscando `license plate detection`.
4. Ejecuta todas las celdas — tardará ~15-30 minutos.
5. Descarga el archivo `best.pt` generado.
6. Colócalo en la carpeta `models/` de tu proyecto local.
7. Convierte a ONNX para mayor rendimiento en CPU:
   ```bash
   python src/export_onnx.py --model best.pt
   ```
8. Actualiza en `.env`: `DETECTION_MODEL=best.onnx` y reinicia el servidor.

---

## ⚙️ Configuración Rápida

Todos los parámetros se modifican en **`config/settings.py`** o en el archivo **`.env`**:

| Variable           | Descripción                                | Default         |
|--------------------|--------------------------------------------|-----------------|
| `DETECTION_MODEL`  | Nombre del modelo en `/models`             | `yolo11n.pt`    |
| `DETECTION_CONF`   | Confianza mínima de detección (0-1)        | `0.4`           |
| `CAMERA_INDEX`     | Índice de cámara (0=integrada, 1=USB)      | `0`             |
| `FLASK_PORT`       | Puerto del servidor Flask                  | `5000`          |

---

## 👥 Equipo y Contribución

Leer **[CONTRIBUTING.md](CONTRIBUTING.md)** antes de hacer tu primer commit. Usamos:
- **Ramas por funcionalidad** (`feature/nombre`, `fix/nombre`)
- **Conventional Commits** (`feat:`, `fix:`, `docs:`, ...)
- **Pull Requests con checklist** hacia `main`

---

## 🛠️ Stack Tecnológico

| Capa           | Tecnología                  |
|----------------|-----------------------------|
| Detección      | Ultralytics YOLO11n         |
| OCR            | EasyOCR                     |
| Backend        | Flask 3.x                   |
| Imagen/Video   | OpenCV                      |
| Inferencia CPU | ONNX Runtime                |
| Entrenamiento  | Google Colab + GPU T4 free  |
| Dataset        | Roboflow                    |
