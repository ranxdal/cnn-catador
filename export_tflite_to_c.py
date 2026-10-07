"""
Convierte un modelo .tflite (ya cuantizado a int8) en un arreglo de C para
integrarlo al firmware del ESP32-CAM con TensorFlow Lite for Microcontrollers
(o con la librería ESP-TFLite / esp-nn de Espressif).

Uso:
    python export_tflite_to_c.py model_out/model_int8.tflite --out model_out/model_data.h
"""
import argparse
from pathlib import Path


def to_c_array(data: bytes, var_name: str) -> str:
    hex_bytes = [f"0x{b:02x}" for b in data]
    lines = []
    for i in range(0, len(hex_bytes), 12):
        lines.append("  " + ", ".join(hex_bytes[i:i + 12]) + ",")
    body = "\n".join(lines)
    return (
        f"// Generado automáticamente a partir de {var_name}.tflite — no editar a mano.\n"
        f"// Tamaño: {len(data)} bytes\n"
        f"#pragma once\n"
        f"#include <cstdint>\n\n"
        f"alignas(8) const unsigned char {var_name}[] = {{\n{body}\n}};\n"
        f"const unsigned int {var_name}_len = {len(data)};\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tflite_path", help="Ruta al modelo .tflite (ya cuantizado a int8)")
    ap.add_argument("--out", default="model_data.h")
    ap.add_argument("--var-name", default="g_cnn_cafe_model")
    args = ap.parse_args()

    data = Path(args.tflite_path).read_bytes()
    header = to_c_array(data, args.var_name)
    Path(args.out).write_text(header)

    print(f"Escrito {args.out} ({len(data)} bytes)")
    print(
        f'En el firmware: #include "{Path(args.out).name}" y usa '
        f"{args.var_name} / {args.var_name}_len para cargar el modelo con "
        "tflite::GetModel(...)."
    )


if __name__ == "__main__":
    main()
