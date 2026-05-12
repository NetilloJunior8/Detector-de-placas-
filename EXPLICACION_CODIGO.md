# Explicación del Código: Detector de Placas (ALPR)

El archivo `src/live_detector.py` contiene el núcleo del sistema de reconocimiento automático de placas vehiculares (ALPR). Su objetivo principal es procesar el video de la cámara en tiempo real, detectar las placas usando un modelo de Inteligencia Artificial (YOLO) y extraer el texto de esas placas mediante Reconocimiento Óptico de Caracteres (OCR).

Lo más importante de este código es su **arquitectura asíncrona y multihilo (multi-threading)**, la cual asegura que el video nunca se congele, incluso si el proceso de leer el texto de la placa se demora.

---

## 1. Estructura General y Arquitectura de Hilos

El sistema divide el trabajo pesado en partes independientes que ocurren de forma paralela. Funciona mediante **3 hilos principales** y **1 ejecutor en segundo plano para el OCR**:

1. **Hilo A (Captura de Video):** `_capture_thread`
   - Se encarga exclusivamente de leer los cuadros (frames) de la cámara web lo más rápido posible, para mantener la fluidez visual (idealmente a 30 FPS).
   - Coloca estos frames en una "cola de entrada" (`_raw_queue`).

2. **Hilo B (Detección YOLO):** `_detection_thread`
   - Toma los frames de la cola de entrada y ejecuta el modelo de inteligencia artificial (YOLO) para buscar los "cuadros delimitadores" (donde está ubicada la placa en la imagen).
   - Envía el trabajo de leer el texto (OCR) a un proceso en segundo plano para que la detección nunca detenga la fluidez del hilo B.
   - Pone el resultado visual en la "cola de renderizado" (`_det_queue`).

3. **Hilo C (Renderizado y Compresión):** `_render_thread`
   - Toma las imágenes con las ubicaciones de las placas de la `_det_queue`.
   - Únicamente codifica o comprime el frame en formato `.jpg` en memoria. No gasta recursos dibujando las cajas sobre la imagen (eso lo hace el frontend/cliente web en HTML5 Canvas con los datos JSON). Esto asegura un enorme ahorro de CPU en el lado del servidor.

4. **Ejecutor Asíncrono OCR (ThreadPoolExecutor):** `_run_ocr_job`
   - Usa la librería de `easyocr` (o en su caso PaddleOCR) para leer la imagen contenida dentro del área que encontró YOLO. Funciona con un solo "trabajador" (worker) para evitar sobrecargar la computadora.
   - Si detecta un texto, pasa por un sistema de validación y **votación** y actualiza el estado general de la aplicación.

---

## 2. Explicación de Funciones Claves

A continuación, la responsabilidad específica de las partes más críticas del archivo:

### Expresiones Regulares y Validación
- `_PLATE_RE`: Es una regla matemática (Expresión Regular) que define cómo se debe ver una placa vehícular de México o LATAM (por ejemplo, tres letras, tres números y una letra `AAA-123-A` o tres letras y tres números `AAA-123`).
- `_is_valid_plate(text)`: Limpia el texto detectado (quitando espacios y guiones) y verifica si encaja en los patrones de `_PLATE_RE`.

### Corrección de Errores Ópticos
- `_correct_plate_characters(text)`: Es común que la IA de OCR confunda caracteres por su similitud visual (Ej: confunde una `O` con un `0` o una `S` con un `5`). Esta función mira el contexto; si el carácter está rodeado de números, asume que debería ser un número (cambia `O` por `0`).

### Clase `ALPRDetector`
Es la clase administradora principal del sistema. En su método de inicialización (`__init__`) "prende los motores": carga el modelo YOLO, la herramienta de OCR y prepara el encolamiento de los procesos.

- **Preprocesamiento Visual (`_preprocess_roi`):**
  Antes de mandar directamente una imagen al OCR, se prepara y mejora para que la IA la entienda mejor. Ocurre lo siguiente:
  1. Se agranda la imagen (Resize).
  2. Pasa a blanco y negro (Escala de Grises).
  3. Se aplica CLAHE para equilibrar el brillo y contraste y combatir zonas con sombras o reflejos.
  4. Se desenfoca un poco (Gaussian Blur) para limpiar "ruido" de la cámara.
  5. Se enfoca intensamente el borde de las letras (Sharpen).
  6. Se convierte en puro blanco y puro negro (Threshold adaptativo), lo que despega las letras totalmente del fondo.

- **Lectura del Texto (`_ocr_box`):**
  Recibe el pequeño cuadro procesado y se lo manda al lector de IA (`self.reader.readtext`). Si no detecta nada aceptable, tiene un "mecanismo de emergencia" (fallback) donde intenta leer sin aplicar todo el filtro blanco y negro agresivo. Termina el proceso pasándolo por el corrector de caracteres.

- **Votación de Placas (`_vote_plate`):**
  Un coche en movimiento nos dará docenas de "lecturas temporales" por las continuas fotos de cámara. Esta función almacena las últimas lecturas. Si una misma lectura es vista `N` cantidad de veces repetidas y coincide con una placa real (`_is_valid_plate`), entonces se declara a esa placa como una "Detección Confirmada".

- **La API del Backend HTTP (`generate_frames`):**
  Es la función que devuelve las imágenes en tiempo real a Flask (la página web) generando un flujo contínuo M-JPEG para proyectar el video fluido directo al navegador web.

---

## 3. ¿Por qué este Código hace el sistema "Posible" y Eficiente?

Este archivo es el **"Corazón" del proyecto** porque coordina diferentes herramientas modernas previniendo que una afecte el funcionamiento de las otras:
1. Las computadoras comunes se bloquean intentando hacer YOLO y OCR al mismo tiempo; la *arquitectura paralela en colas de este código* evita exactamente ese problema.
2. Emplea tácticas para corregir la perspectiva de entornos sucios con *procesamiento matemático de imagen robusto (CLAHE + AdaptativeThreshold)*.
3. No da una alerta con la primera coincidencia que lee, sino que tiene un sistema de votación para otorgar una placa solo con altísima certidumbre en las respuestas obtenidas del modelo de inteligencia.
4. Todo opera de lado del backend, delegando en este sistema el cálculo pesado sin gastar recursos haciendo interfaces visuales incrustadas en el video original.
