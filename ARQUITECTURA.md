# Arquitectura del Sistema ALPR (Automatic License Plate Recognition)

Este documento explica de forma detallada cómo está construido el sistema de detección de placas, el "por qué" de las decisiones técnicas empleadas y el "cómo" funciona su flujo de datos en tiempo real. Es la guía definitiva para la presentación de la arquitectura.

---

## 1. El Propósito del Sistema (El Porqué)
El objetivo de la aplicación es visualizar video en tiempo real desde una cámara y extraer el número de placa de los vehículos, logrando una **alta precisión de lectura** sin comprometer la **fluidez del video**.

El mayor reto en la Visión Computacional (Computer Vision) en tiempo real es que los algoritmos de Inteligencia Artificial (Especialmente OCR - Reconocimiento Óptico de Caracteres) son matemáticamente pesados. Si intentamos **1) Leer de la cámara**, **2) Buscar vehículos**, y **3) Leer el texto** en un solo ciclo secuencial, el video se vería "congelado" (a 1 o 2 cuadros por segundo). Para evitar este cuello de botella y ofrecer una experiencia fluida que fluya a la velocidad del video natural, el sistema se diseñó de cero bajo un enfoque de paralelismo estricto.

---

## 2. Tecnologías Principales (Las Herramientas)
- **OpenCV**: Maneja la conexión directa a la cámara web, el procesamiento numérico de imágenes (niveles de contraste, escala de grises) y la compresión de video (MJPEG).
- **YOLO (Ultralytics)**: Es el modelo neuronal de detección de objetos. Está entrenado exclusivamente para responder: *¿Dónde hay una placa en esta imagen?*. Retorna rápidamente las coordenadas (Bounding Boxes) de la matrícula.
- **EasyOCR**: Es el motor de Reconocimiento Óptico de Caracteres. Se encarga de traducir los píxeles de la región donde YOLO dijo que había una placa en texto plano.
- **Flask**: El framework web moderno que emite el video procesado y responde las solicitudes en formato ligero al navegador web del usuario sin la carga de renderizado nativo.
- **SQLite**: Motor de base de datos integrado que, a través de la clase `PlateLogger`, persista permanentemente en un archivo local las placas confirmadas.

---

## 3. Arquitectura del Pipeline: Multi-Hilo (El Cómo)
La clave del rendimiento impecable radica en la distribución asíncrona de recursos en la CPU/GPU. El archivo principal de procesamiento (`live_detector.py`) se fragmenta en **3 Hilos Principales (Threads)** y **1 Ejecutor Asíncrono**.

### Hilo A: Captura Continua (Zero Lag)
Su trabajo es exclusivamente extraer *frames* de la cámara abierta (ej. a 30 FPS constantes) mediante OpenCV y amontonarlos en una pequeña "sala de espera" temporal (`_raw_queue`). Si OpenCV saca un frame nuevo y la sala está llena, descarta el viejo garantizando que **siempre tengamos la imagen más reciente posible**, evadiendo desfasamientos de la vida real.

### Hilo B: Inferencia Múltiple (Detección YOLO)
1. Toma el frame más reciente y lo pasa por el ojo de **YOLO**.
2. YOLO se ejecuta cada `INFER_EVERY_N` frames para no acaparar toda la computadora. Al analizar, YOLO encuentra la placa y encierra la región pero **NO se detiene a leer el texto**. Ya que "detectar" es hasta 10 veces más veloz que "leer".
3. **Desvío Asíncrono**: Despacha la pequeña imagen recortada a una zona apartada (El OCR Executor) para que trabaje sin congelar la imagen, y el Hilo B continúa su propio trabajo ágilmente de inmediato.

### Executor OCR: El Operario Analítico
El executor ocurre completamente detrás del escenario de forma paralela. Solamente recibe *Recortes de la imagen original* de la placa.
Aquí entran las lógicas especializadas de filtrado de ruido para convertir correctamente la imagen en texto. Una vez que descifra el texto, actualiza una "Memoria Compartida Global" con el nuevo dato y desaparece, todo el ciclo toma algunas fracciones de segundo, pero debido al paralelismo, la cámara del usuario jamás percibe que se interrumpió la grabación.

### Hilo C: Renderización de Transmisión
Se encarga únicamente de empaquetar en tiempo real los cuadros que salen de la arquitectura (codificándolos en JPG muy rápidamente) y emitirlos vía red para la visualización del usuario del frontend en HTML/Javascript. Delega la creación de cuadros y de etiquetas virtuales al navegador, logrando eficiencia pura en el lado del servidor.

---

## 4. El Cerebro Antirruido: Pipeline del OCR
Para combatir reflejos, sombras, formatos complejos de cámaras de seguridad y texto basura alrededor de la placa estatal (como años o nombres de Estado), el sistema aplica una secuencia de refinamiento robusta antes de decidir cuál es la placa final.

1. **Ajuste y Realce**: La imagen recortada se somete a escalas de grises, aumento dramático de contrastes por adaptaciones zonales (CLAHE), y aumento de nitidez, revelando las sombras difíciles.
2. **Lista Blanca (Allowlist)**: El OCR está restringido de fábrica para solamente ver **`A-Z y 0-9`**, previniendo que alucine caracteres inválidos que detonen errores lógicos (`, @ / ?`).
3. **Filtro Estructural de Alturas**: El sistema analiza geométricamente todas las formaciones de texto leídas en la región. Mide la altura de cada bloque. Solo retiene el texto que equivale al menos al 55% de la altura más prominente. Esto asegura el descarte total de caracteres "residuales" pequeños ("Estado de México", "Gto", "20"), quedando puramente el bloque alfanumérico principal.
4. **Votación de Memoria Corta (Pooling Engine)**: Porque **un frame puede fallar pero no un movimiento**, la lectura resultante es anexada a un histórico fugaz en movimiento. Solo cuando la inteligencia logra **converger `X` veces en la misma lectura literal exacta** (`Votación Mínima`), se declara la placa como **Confirmada** oficial del sistema. Se evalúa con RegEx y finalmente se almacena de forma persistente en Base de Datos.
