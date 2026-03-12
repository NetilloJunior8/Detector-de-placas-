"""
scripts/export_onnx_opset17.py
-------------------------------
Re-exporta best.pt → best.onnx con opset=17 compatible con onnxruntime 1.x.

El modelo exportado por defecto desde Colab usa opset 22 (ultralytics 8.4.21),
pero onnxruntime solo garantiza soporte hasta opset 21.

Uso:
    python scripts/export_onnx_opset17.py --pt models/best.pt
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", default="models/best.pt", help="Ruta al archivo .pt")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset (default: 17)")
    args = parser.parse_args()

    pt_path = PROJECT_ROOT / args.pt
    if not pt_path.exists():
        print(f"❌ No se encontró el archivo: {pt_path}")
        print("   Descarga best.pt desde Google Drive y ponlo en models/")
        sys.exit(1)

    print(f"📦 Cargando modelo: {pt_path}")
    model = YOLO(str(pt_path))

    print(f"🔄 Exportando a ONNX con opset={args.opset}...")
    output = model.export(
        format="onnx",
        imgsz=640,
        dynamic=False,
        opset=args.opset,
        simplify=False,   # deshabilitar onnxslim para evitar crashes en Windows
    )

    # Mover el .onnx exportado a models/best.onnx
    output_path = Path(str(output))
    dest = PROJECT_ROOT / "models" / "best.onnx"
    output_path.replace(dest)

    print(f"✅ Exportado correctamente: {dest}")
    print(f"   Opset: {args.opset}")

if __name__ == "__main__":
    main()
