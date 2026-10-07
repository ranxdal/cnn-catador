"""
Captura y etiqueta fotos REALES de grano de café con la webcam, para la etapa
de ENTRENAMIENTO (la etapa en la que nosotros, como personas, preparamos las
muestras a propósito y sabemos con certeza a qué categoría pertenece cada
una). El sistema embebido nunca hace esto — esto es exclusivamente el paso
manual de recolección de datos antes de entrenar.

Flujo pensado para este script:
  1. Preparan físicamente una muestra de una categoría conocida (por ejemplo,
     una porción que dejaron sobretostar a propósito).
  2. Corren este script, seleccionan la tecla de esa categoría y toman varias
     fotos de esa misma muestra (distintos granos, ligeros cambios de
     ángulo/posición) — todas quedan etiquetadas con esa categoría.
  3. Repiten para cada una de las otras tres muestras preparadas.

Categorías de calidad (coinciden con las definidas en el documento del
proyecto: "calidad alta, rancio, con humedad o quemado"):
  [1] calidad_alta  — muestra fresca, tueste correcto
  [2] quemado       — muestra sobretostada a propósito
  [3] humedad       — muestra expuesta a humedad a propósito
  [4] rancio        — muestra dejada ponerse rancia / con moho a propósito

Clase de rechazo (NO es una categoría de calidad, es para que el modelo
aprenda a distinguir "esto no es café" de una muestra real):
  [5] no_cafe       — cualquier cosa que NO sea grano de café: la mano,
                      la mesa vacía, otros granos/semillas, objetos random.
                      Mientras más variado, mejor aprende a rechazar.

Controles durante la captura:
  1 / 2 / 3 / 4  -> guarda el frame actual en la categoría correspondiente
  q / ESC        -> salir

Si todavía no tienen las cuatro muestras físicas listas (por ejemplo, la
"rancio" tarda varios días en prepararse), no hace falta esperar: usen
--labels para activar solo las categorías que sí tienen hoy. Las demás
carpetas simplemente no se tocan en esta sesión, y pueden correr el script
de nuevo otro día para completarlas — train_cnn.py toma las clases de las
carpetas que existan, así que no hay que rehacer nada.

    # Solo calidad_alta y quemado por hoy:
    python capture_webcam.py --out dataset --labels 1=calidad_alta,2=quemado

    # Más adelante, cuando ya tengan las cuatro muestras:
    python capture_webcam.py --out dataset

Requisitos:
    pip install opencv-python

Uso:
    python capture_webcam.py --out dataset --cam 0
"""
import argparse
import time
from pathlib import Path

import cv2

IMG_SIZE = 96  # debe coincidir con IMG_SIZE en train_cnn.py

DEFAULT_CLASSES = {
    "1": "calidad_alta",
    "2": "quemado",
    "3": "humedad",
    "4": "rancio",
    "5": "no_cafe",  # clase de rechazo: NO es una categoría de calidad, es para
                     # que el modelo aprenda a reconocer que algo no es café
                     # (mano, mesa vacía, otro grano, objetos random, etc.)
}


def parse_labels(labels_arg):
    """Convierte '1=calidad_alta,2=quemado' en {ord('1'): 'calidad_alta', ...}.
    Sin --labels, usa las cuatro categorías por defecto."""
    if not labels_arg:
        return {ord(k): v for k, v in DEFAULT_CLASSES.items()}

    result = {}
    for pair in labels_arg.split(","):
        key, name = pair.split("=")
        key = key.strip()
        name = name.strip()
        if len(key) != 1:
            raise SystemExit(f"Cada tecla debe ser un solo carácter, revisa: '{pair}'")
        result[ord(key)] = name
    return result


def center_square_crop(frame):
    """Recorta al cuadrado central en la resolución NATIVA de la cámara (sin
    reducir todavía) — esto es lo que se muestra en pantalla, para que se vea
    nítido. El reducido a IMG_SIZE se hace aparte, solo al guardar."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side]


def to_model_size(square_crop):
    """Reduce el recorte cuadrado a IMG_SIZE x IMG_SIZE — esta es la versión
    que se guarda en disco y la que el modelo espera."""
    return cv2.resize(square_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--cam", type=int, default=0, help="Índice de la webcam (0 suele ser la integrada)")
    ap.add_argument("--val-every", type=int, default=5,
                     help="De cada N fotos guardadas por categoría, 1 va a val en vez de train")
    ap.add_argument("--labels", default=None,
                     help="Categorías activas en esta sesión, ej: '1=calidad_alta,2=quemado'. "
                          "Sin esta opción, se activan las cuatro por defecto.")
    args = ap.parse_args()

    CLASSES = parse_labels(args.labels)

    out = Path(args.out)
    counts = {name: {"train": 0, "val": 0} for name in CLASSES.values()}
    for name in CLASSES.values():
        for split in ("train", "val"):
            (out / split / name).mkdir(parents=True, exist_ok=True)
            counts[name][split] = len(list((out / split / name).glob("*.png")))

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la webcam con índice {args.cam}")
    # Pide la resolución nativa más alta disponible; si la cámara no la
    # soporta, simplemente se queda con su máximo (esto no arregla el
    # enfoque, pero evita usar un modo de vista previa innecesariamente bajo).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("Preparen la primera muestra y presionen la tecla de su categoría.")
    print("  " + "   ".join(f"{chr(k)} = {n}" for k, n in CLASSES.items()) + "   q = salir\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer un frame de la cámara.")
            break

        square = center_square_crop(frame)
        display = cv2.resize(square, (384, 384), interpolation=cv2.INTER_LINEAR)

        total_by_class = {n: counts[n]["train"] + counts[n]["val"] for n in CLASSES.values()}
        y = 24
        for key, name in CLASSES.items():
            text = f"[{chr(key)}] {name}: {total_by_class[name]} fotos"
            cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            y += 22

        cv2.imshow("Captura de muestras - Catador Digital", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):  # 27 = ESC
            break

        if key in CLASSES:
            name = CLASSES[key]
            # cada --val-every fotos de esta categoría, una va a val
            n_total = total_by_class[name]
            split = "val" if (n_total + 1) % args.val_every == 0 else "train"
            fname = out / split / name / f"{name}_{int(time.time() * 1000)}.png"
            cv2.imwrite(str(fname), to_model_size(square))
            counts[name][split] += 1
            print(f"Guardada: {fname}  (train={counts[name]['train']}, val={counts[name]['val']})")

    cap.release()
    cv2.destroyAllWindows()

    print("\nResumen de la sesión de captura:")
    for name in CLASSES.values():
        print(f"  {name}: train={counts[name]['train']}  val={counts[name]['val']}")
    print(
        "\nRecuerden: cada foto quedó guardada bajo la categoría que ustedes "
        "seleccionaron a propósito, sabiendo con certeza qué muestra era. "
        "Repitan esta sesión con distintos granos/ángulos por cada muestra "
        "hasta tener suficientes fotos por categoría antes de correr "
        "train_cnn.py."
    )


if __name__ == "__main__":
    main()