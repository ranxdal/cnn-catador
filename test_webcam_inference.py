"""
Prueba en vivo, con la webcam, el modelo YA ENTRENADO — esta es la etapa de
USO: nadie etiqueta nada aquí, el sistema clasifica la muestra por sí mismo,
tal como lo haría el ESP32-CAM una vez que el modelo se integre al firmware.
Sirve para validar el reconocimiento antes de tener el dispositivo ESP32
armado.

Requisitos:
    pip install opencv-python tensorflow numpy

Uso (modelo Keras, más simple, recomendado mientras solo prueban en la PC):
    python test_webcam_inference.py --model model_out/cnn_cafe.h5 --classes model_out/class_names.txt

Uso (modelo .tflite cuantizado, para verificar que el modelo que SÍ va al
ESP32-CAM sigue clasificando bien después de la cuantización):
    python test_webcam_inference.py --model model_out/model_int8.tflite --classes model_out/class_names.txt --tflite
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 96  # debe coincidir con IMG_SIZE en train_cnn.py / capture_webcam.py


def center_square_crop(frame):
    """Recorta al cuadrado central en resolución nativa (para mostrar nítido)."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side]


def to_model_size(square_crop):
    """Reduce a IMG_SIZE x IMG_SIZE — el tamaño que espera el modelo."""
    return cv2.resize(square_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def load_keras_predictor(model_path):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)

    def predict(bgr_crop):
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        batch = np.expand_dims(rgb, axis=0)
        return model.predict(batch, verbose=0)[0]

    return predict


def load_tflite_predictor(model_path):
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    in_scale, in_zero = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    def predict(bgr_crop):
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        quantized = (rgb / in_scale + in_zero).astype(input_details["dtype"])
        batch = np.expand_dims(quantized, axis=0)
        interpreter.set_tensor(input_details["index"], batch)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_details["index"])[0].astype("float32")
        if out_scale:  # de vuelta a probabilidades aproximadas en [0, 1]
            raw = (raw - out_zero) * out_scale
        return raw

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ruta a cnn_cafe.h5 o a model_int8.tflite")
    ap.add_argument("--classes", required=True, help="Ruta a class_names.txt (generado por train_cnn.py)")
    ap.add_argument("--tflite", action="store_true", help="Indica que --model es un .tflite cuantizado")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--min-confidence", type=float, default=0.55,
                     help="Si la confianza de la clase top queda por debajo de este valor, "
                          "se muestra 'No reconocido' en vez de forzar una categoría (0.0-1.0).")
    args = ap.parse_args()

    class_names = Path(args.classes).read_text().strip().splitlines()
    predict = load_tflite_predictor(args.model) if args.tflite else load_keras_predictor(args.model)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la webcam con índice {args.cam}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("Mostrando clasificación en vivo. El sistema decide solo — nadie etiqueta nada aquí.")
    print("Presiona q o ESC para salir.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        square = center_square_crop(frame)
        probs = predict(to_model_size(square))
        top_idx = int(np.argmax(probs))
        top_label = class_names[top_idx] if top_idx < len(class_names) else f"clase_{top_idx}"
        top_conf = float(probs[top_idx])

        # Si el modelo no está seguro de ninguna clase conocida (o si la clase
        # top es la de rechazo "no_cafe"), no se fuerza una categoría de calidad.
        is_rejected = top_conf < args.min_confidence or top_label == "no_cafe"

        display = cv2.resize(square, (420, 420), interpolation=cv2.INTER_LINEAR)
        if is_rejected:
            shown_text = f"No reconocido como cafe  ({top_conf * 100:.0f}%)"
            color = (0, 0, 255)  # rojo (BGR)
        else:
            shown_text = f"{top_label}  ({top_conf * 100:.0f}%)"
            color = (0, 255, 0)  # verde
        cv2.putText(display, shown_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        y = 60
        for i, name in enumerate(class_names):
            p = float(probs[i]) if i < len(probs) else 0.0
            bar_w = int(p * 250)
            cv2.rectangle(display, (10, y), (10 + bar_w, y + 14), (0, 200, 0), -1)
            cv2.rectangle(display, (10, y), (260, y + 14), (255, 255, 255), 1)
            cv2.putText(display, name, (270, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22

        cv2.imshow("Clasificacion en vivo - Catador Digital (etapa de uso)", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()