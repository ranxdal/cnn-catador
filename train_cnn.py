"""
Entrena una CNN ligera para clasificar la calidad visual del grano de café
tostado a partir de imágenes capturadas con la cámara del ESP32-CAM, y la
exporta a TensorFlow Lite (float32 e int8 cuantizado) lista para integrarse
al firmware con TensorFlow Lite for Microcontrollers.

Requisitos:
    pip install tensorflow pillow numpy

Uso:
    python train_cnn.py --data dataset --epochs 20 --out model_out
"""
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 96           # debe coincidir con generate_synthetic_data.py y con el
                         # recorte de imagen que entregue el firmware desde la OV2640
BATCH_SIZE = 16


def build_datasets(data_dir: Path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )
    class_names = train_ds.class_names

    normalize = layers.Rescaling(1.0 / 255)
    train_ds_norm = train_ds.map(lambda x, y: (normalize(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds_norm = val_ds.map(lambda x, y: (normalize(x), y)).prefetch(tf.data.AUTOTUNE)
    return train_ds_norm, val_ds_norm, class_names


def build_model(num_classes: int) -> tf.keras.Model:
    # Arquitectura pequeña a propósito: debe caber cuantizada a int8 dentro de
    # la memoria limitada del ESP32-CAM (idealmente el .tflite < 200-300 KB).
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)

    x = layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)                                  # 48x48

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)                                  # 24x24

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)                                  # 12x12

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(24, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def representative_dataset_gen(train_ds, num_samples=100):
    def gen():
        count = 0
        for images, _ in train_ds.unbatch().batch(1):
            yield [images]
            count += 1
            if count >= num_samples:
                break
    return gen


def export_tflite(model, train_ds, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) versión float32, para verificar en PC que la conversión no rompe nada
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float = converter.convert()
    (out_dir / "model_float32.tflite").write_bytes(tflite_float)

    # 2) versión cuantizada int8, la que se integra al ESP32-CAM
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(train_ds)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8 = converter.convert()
    (out_dir / "model_int8.tflite").write_bytes(tflite_int8)

    print(f"\nmodel_float32.tflite: {len(tflite_float) / 1024:.1f} KB")
    print(f"model_int8.tflite:    {len(tflite_int8) / 1024:.1f} KB  <- esta es la que va al firmware")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="model_out")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = build_datasets(data_dir)
    print("Clases detectadas (en este orden, el índice = salida del modelo):", class_names)
    (out_dir / "class_names.txt").write_text("\n".join(class_names))

    model = build_model(num_classes=len(class_names))
    model.summary()

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\nPrecisión de validación: {val_acc * 100:.1f}%")

    model.save(out_dir / "cnn_cafe.h5")
    export_tflite(model, train_ds, out_dir)

    print(
        f"\nListo. Revisa {out_dir}/ — usa model_int8.tflite con "
        "export_tflite_to_c.py para generar el arreglo en C que se carga en "
        "el firmware del ESP32-CAM."
    )


if __name__ == "__main__":
    main()
