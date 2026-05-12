# Reporte Técnico de Práctica: Sistema Integrado de Reconocimiento Automático de Matrículas (ALPR)

## 1. Introducción
El presente reporte documenta el diseño, desarrollo e implementación de un sistema de Reconocimiento Automático de Matrículas (ALPR, por sus siglas en inglés) en tiempo real. La necesidad de identificar vehículos de manera automatizada es crucial en sistemas de peaje, control de accesos, seguridad pública y gestión de estacionamientos. Este proyecto aborda el desafío tecnológico de extraer texto (placas vehiculares) a partir de un flujo de video continuo (cámara web) solucionando los clásicos cuellos de botella de procesamiento computacional que afectan a los sistemas tradicionales.

## 2. Objetivos

### Objetivo General
Desarrollar e implementar un sistema ALPR funcional de extremo a extremo que permita detectar y reconocer placas vehiculares latinoamericanas en tiempo real mediante visión por computadora y aprendizaje profundo, desplegando los resultados a través de una interfaz web con nulo impacto en la fluidez del video.

### Objetivos Específicos
1. Entrenar y desplegar un modelo de detección de objetos ligero (**YOLO11n**) para aislar la región ocupada por la placa vehicular.
2. Implementar un motor de Reconocimiento Óptico de Caracteres (**OCR**) con técnicas de preprocesamiento de imágenes avanzadas para asegurar altas tasas de acierto.
3. Diseñar una **arquitectura multi-hilo (multi-threading) asíncrona** que aísle la captura de video del procesamiento OCR intensivo.
4. Desarrollar un backend web con **Flask** para servir el procesamiento vía red al navegador.
5. Persistir las placas válidas y confirmadas en una base de datos local **SQLite** para su control histórico.

---

## 3. Tecnologías y Herramientas Utilizadas (Stack Tecnológico)

El sistema hace uso de herramientas vanguardistas en las áreas de Inteligencia Artificial y Desarrollo Web:

*   **Visión Computacional y Preprocesamiento:** `OpenCV`. Se utilizó para el enrutamiento de la cámara web, la aplicación de filtros morfologicos (CLAHE, Umbralización Adaptativa) y la codificación MJPEG del video en vivo.
*   **Detección de Objetos:** `YOLO11n` (Ultralytics). Modelo seleccionado por su bajo peso (Nano) y alta velocidad de inferencia, ideal para hardware de pocos recursos (como laptops o CPUs sin gráficas dedicadas). Fue transformado a formato ONNX (`ONNX Runtime`) para mayor velocidad de ejecución.
*   **Reconocimiento de Texto (OCR):** `EasyOCR` / `PaddleOCR`. Motores de aprendizaje profundo especializados en leer el texto dentro de la región recortada de la placa (Bounding Box).
*   **Backend y Servidor Web:** `Flask 3.x` framework de Python elegido por su ligereza y facilidad de implementar streaming continuo de video (Generadores de frames).
*   **Entorno de Entrenamiento MLOps:** `Google Colab (GPU T4)` y la plataforma de datasets `Roboflow`, para compilar el dataset y entrenar el modelo en la nube salvaguardando los recursos locales.
*   **Almacenamiento:** Base de datos relacional ligera integrada vía `SQLite` (`PlateLogger`).

---

## 4. Arquitectura y Desarrollo del Sistema

### 4.1. Diseño Multi-Hilo Asíncrono (El núcleo de rendimiento)
El obstáculo principal en el desarrollo de un sistema ALPR es el cuello de botella (latencia): leer un texto con IA neuronal normalmente bloquea el ciclo de programa y congela el video a 1 o 2 FPS. Para solucionarlo, se desarrolló una arquitectura concurrente contenida en `src/live_detector.py` basada en múltiples hilos (Threads) y colas de mensajes (`Queue`):

1.  **Hilo A (Captura Continua - Zero Lag):** Dedicado exclusivamente a vaciar la memoria de la cámara web mediante OpenCV. Siempre guarda el frame más reciente y descarta los viejos instantáneamente. Así se asegura el "Zero Lag" entre el entorno físico real y la imagen leída.
2.  **Hilo B (Detección YOLO):** Periódicamente toma un frame y lo analiza con la red YOLO11n. Si encuentra la placa, recorta la región pero **NO** se detiene a leer el texto. En su lugar, envía este pequeño recorte a un proceso oculto en segundo plano (Pool Asíncrono OCR).
3.  **Ejecutor Asíncrono (OCR):** Trabaja silenciosamente; recibe únicamente el parche de imagen, realiza los filtros, extrae el texto, y luego actualiza la base de datos y la memoria general. El Hilo B jamás lo espera, por lo que el escaneo visual no se interrumpe.
4.  **Hilo C (Renderización y Red):** Codifica las transacciones en JPG y las manda vía red HTTP hacia el framework Flask sin gastar procesamiento dibujando recuadros, delegando esa tarea gráfica al cliente HTML5 final.

### 4.2. Algoritmo Antirruido y Preprocesamiento Matemático
Para que el motor OCR no falle ante sombras duras, ángulos complejos o destellos de faros, el software aplica un pipeline sistemático de limpieza antes de intentar leer:
*   Paso de arreglo de Color BGR a Escala de Grises.
*   **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Un ecualizador para uniformizar los desbalances lumínicos en placas oscurecidas.
*   Filtro de Afilado (Sharpening Kernel) combinado con Desenfocado Gaussiano (Blur) para reducir ruido de sensor.
*   **Binarización (Adaptive Threshold):** Transforma la placa a puro blanco y puro negro despegando dramáticamente el texto (negro) de los fondos y texturas del estado.

### 4.3. Lógica de Negocio y Sistema de Votación
El programa no aprueba la primera lectura recibida al azar. Implementa una "Votación de Memoria Corta":
*   **Filtros Geométricos:** Se descartan letras leyendo su altura. Letras y números residuales promocionales (como el estado "Gto", o el año) miden mucho menos que la fila central de letras y números alfanuméricos, por lo tanto se suprimen.
*   **Expresiones Regulares (Regex):** Toda lectura posible pasa por un validador matemático estricto que requiere un formato oficial, ejemplo: `AAA-123-A` o `AAA-123`.
*   **Pooling Engine:** Un coche en movimiento es analizado varias veces de corrido. Únicamente cuando la IA y la Regex convergen y dictan ***la misma lectura exacta `N` cantidad de veces consecutivas***, la placa se cuenta como "Confirmada Exitosamente" y evita registrar repetidas de la misma.

---

## 5. Metodología de MLOps y Entrenamiento
La creación de la inteligencia artificial requirió la recopilación de datos exhaustiva y entrenamiento en supercómputo:
1.  **Recolección de Datos:** Se obtuvo, analizó y etiquetó un conjunto de datos extenso de placas vehiculares procedentes de _Roboflow Universe_.
2.  **Entrenamiento en Nube:** Mediante Jupyter Notebooks ejecutados sobre instancias gratuitas en **Google Colab** utilizando aceleración de hardware (T4 GPU). El modelo se entrenó sobre más de 100 épocas.
3.  **Optimización ONNX:** Los pesos originales del modelo `best.pt` arrojaban un uso de CPU y RAM considerable. Se transcodificó hacia `.onnx`. Este formato agiliza drásticamente el cálculo multimatricial posibilitando arrancar IA en tiempo real sin tarjeta de video dedicada.

---

## 6. Resultados y Rendimiento Observado
Bajo escenarios de prueba simulada y exposición a cámara real, los resultados fueron sustancialmente eficientes:
*   **Fluidez Mantenida:** Gracias a la separación asíncrona entre el Hilo B (Detección) y el Ejecutor (OCR), jamás se experimentó congelamiento de cámara, manteniendo una tasa de actualización fluida rondando los 30 cuadros por segundo de ida al navegador del usuario.
*   **Tasa de Disparo OCR:** El uso preprocesamiento matemático (CLAHE/Binarización) disparó los aciertos del modelo PaddleOCR/EasyOCR permitiéndole encontrar letras y números incluso bajo sombras pronunciadas en la chapa del vehículo.
*   **Cero Falsos Positivos:** El sistema cruzado de Regex sumado al esquema de votaciones consecutivas erradicaron exitosamente las lecturas basura provenientes de stickers en el chasis de los coches o el fondo de los panoramas urbanos.

---

## 7. Conclusiones del Proyecto
El desarrollo y arquitectura de este Sistema ALPR demostró contundentemente que **la formulación de una lógica de software asíncrona inteligente puede suplir satisfactoriamente las carencias del hardware**. Frecuentemente, el análisis IA aplicado de manera secuencial torna este tipo de sistemas totalmente inoperables en computadoras estándar. No obstante, al explotar el multithreading Python interconectado al poder de YOLO11n y las matemáticas eficientes de OpenCV, ha resultado en una solución que corre en tiempo real con latencias ínfimas. El desarrollo final constituye una pieza de software limpia, escalable y una valiosa práctica integradora de Inteligencia Artificial conectada a los servicios web.
