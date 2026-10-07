"""
Genera un dataset SINTÉTICO de imágenes de granos de café tostado para
validar el pipeline completo de la CNN (entrenar -> exportar -> integrar)
sin necesidad de fotos reales todavía.

Ahora que cuentan con una webcam, lo normal es capturar fotos reales con
capture_webcam.py en lugar de usar este generador — este script sigue siendo
útil solo como una prueba rápida de "¿el pipeline completo corre de punta a
punta?" antes de invertir tiempo en preparar las muestras físicas.

Categorías (coinciden con las definidas en el documento del proyecto:
"calidad alta, rancio, con humedad o quemado"), pensadas como desviaciones
respecto al café arábica de tueste medio que usan como base, no como niveles
de tueste:

  - calidad_alta : muestra fresca, tueste correcto, color marrón consistente
  - quemado      : muestra sobretostada, manchas negras, alto contraste
  - humedad      : muestra expuesta a humedad, tono apagado/grisáceo,
                    apariencia "hinchada" y con brillo desigual
  - rancio       : muestra rancia o con moho, color irregular,
                    manchas verdes/blancas

Uso:
    python generate_synthetic_data.py --out dataset --per-class 300
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IMG_SIZE = 96  # debe coincidir con IMG_SIZE en train_cnn.py

CLASS_PARAMS = {
    "calidad_alta": {
        "base_color": (120, 78, 45),   # marrón medio (tueste correcto)
        "color_jitter": 12,
        "spot_prob": 0.05,
        "spot_color_range": ((90, 60, 35), (140, 95, 55)),
    },
    "quemado": {
        "base_color": (55, 35, 22),    # marrón muy oscuro / casi negro
        "color_jitter": 18,
        "spot_prob": 0.25,
        "spot_color_range": ((10, 8, 8), (40, 25, 18)),
    },
    "humedad": {
        "base_color": (150, 130, 100),  # tono apagado / grisáceo, aspecto húmedo
        "color_jitter": 15,
        "spot_prob": 0.15,
        "spot_color_range": ((80, 70, 55), (110, 95, 75)),  # manchas de humedad
    },
    "rancio": {
        "base_color": (140, 120, 60),  # tono verdoso / grano rancio
        "color_jitter": 30,
        "spot_prob": 0.30,
        "spot_color_range": ((60, 90, 40), (210, 220, 190)),  # verde y moho
    },
}


def _clamp(v):
    return max(0, min(255, int(v)))


def make_bean_image(base_color, color_jitter, spot_prob, spot_color_range, size=IMG_SIZE):
    img = Image.new("RGB", (size, size), (30, 24, 20))  # fondo (mesa/superficie)
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    rx, ry = size * 0.38, size * 0.30
    for y in range(size):
        for x in range(size):
            dx, dy = (x - cx) / rx, (y - cy) / ry
            if dx * dx + dy * dy <= 1.0:
                jitter = random.randint(-color_jitter, color_jitter)
                r = _clamp(base_color[0] + jitter)
                g = _clamp(base_color[1] + jitter)
                b = _clamp(base_color[2] + jitter)
                img.putpixel((x, y), (r, g, b))

    # surco central típico del grano de café
    draw.line([(cx, cy - ry * 0.7), (cx, cy + ry * 0.7)], fill=(20, 15, 12), width=2)

    # manchas (defectos, moho, carbonización) según la clase
    n_spots = int(spot_prob * 40)
    for _ in range(n_spots):
        sx = random.randint(int(cx - rx * 0.8), int(cx + rx * 0.8))
        sy = random.randint(int(cy - ry * 0.8), int(cy + ry * 0.8))
        r = random.randint(2, 6)
        c = tuple(random.randint(spot_color_range[0][i], spot_color_range[1][i]) for i in range(3))
        draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=c)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    n_val = int(args.per_class * args.val_split)

    for cls, params in CLASS_PARAMS.items():
        for split, count in (("train", args.per_class - n_val), ("val", n_val)):
            split_dir = out / split / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                img = make_bean_image(
                    params["base_color"],
                    params["color_jitter"],
                    params["spot_prob"],
                    params["spot_color_range"],
                )
                img.save(split_dir / f"{cls}_{split}_{i:04d}.png")
            print(f"{cls}/{split}: {count} imágenes generadas en {split_dir}")

    print(
        "\nListo. Esto es SOLO para validar el pipeline (entrenar -> exportar -> "
        "integrar), igual que se hizo con el modelo Random Forest.\n"
        "Como ya cuentan con una webcam, lo recomendable es reemplazar estas "
        "carpetas con fotos reales capturadas con capture_webcam.py, manteniendo "
        "la misma estructura:\n"
        "  dataset/train/<clase>/*.png\n"
        "  dataset/val/<clase>/*.png"
    )


if __name__ == "__main__":
    main()
