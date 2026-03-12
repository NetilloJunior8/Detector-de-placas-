# Guía de Contribución — Proyecto ALPR

¡Gracias por contribuir! Sigan estas normas para que el equipo trabaje sin conflictos.

## Flujo de Trabajo con Git (Git Flow simplificado)

La rama `main` es estable y contiene el código que **funciona comprobado**. Nunca hagan push directo a `main`.

```bash
# 1. Siempre partir de main actualizado
git checkout main
git pull origin main

# 2. Crear su propia rama descriptiva  
git checkout -b feature/nombre-descriptivo
#  Ejemplo de nombres válidos:
#  feature/tracking-bytesort
#  fix/ocr-preprocesamiento
#  docs/agregar-guia-dataset

# 3. Hacer sus cambios y commits pequeños/descriptivos
git add .
git commit -m "feat: agregar pre-procesamiento de perspectiva al OCR"

# 4. Subir su rama
git push origin feature/nombre-descriptivo

# 5. Abrir un Pull Request hacia main en GitHub y pedir revisión
```

## Convención para Mensajes de Commit

Usar prefijos estándar (Conventional Commits):

| Prefijo      | Cuándo usarlo                                 |
|--------------|-----------------------------------------------|
| `feat:`      | Agrega una nueva funcionalidad                 |
| `fix:`       | Corrige un bug                                 |
| `docs:`      | Cambios solo en documentación                 |
| `refactor:`  | Mejora de código sin cambiar comportamiento   |
| `test:`      | Agrega o modifica pruebas                      |
| `chore:`     | Cambios en config, .gitignore, etc.           |

## Reglas de Calidad

1. **Nunca subir modelos (`.pt`, `.onnx`)** — Son pesados y están en `.gitignore`. Compartirlos por Google Drive o Discord.
2. **Nunca subir el `.env`** — Usar `.env.example` como referencia.
3. **Probar localmente antes** de hacer Pull Request.
4. **Instalar dependencias desde `requirements.txt`** — Si agregan un paquete nuevo, actualizar ese archivo.
5. **Un Pull Request por tema/funcionalidad** — No mezclar fixes con features.

## ¿Cómo configuro mi entorno?

```bash
git clone https://github.com/tu-usuario/ProyectoPlacasV1.git
cd ProyectoPlacasV1
python -m venv venv
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env      # Rellenar con tus datos locales
```

Luego corre el servidor:
```bash
python src/app.py
```
Y abre `http://localhost:5000` en tu navegador.
