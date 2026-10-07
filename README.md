# CNN — Catador Digital

Pipeline de la parte de visión del proyecto: clasifica el grano de café
tostado a partir de una imagen, en paralelo al modelo Random Forest que
analiza aroma + temperatura/humedad. Por ahora las pruebas se hacen con una
webcam (no cuentan todavía con el ESP32 armado); todo este pipeline está
pensado para portarse tal cual al ESP32-CAM cuando esté disponible.

## Dos etapas — no confundir

**1. Entrenamiento (lo hacen ustedes, como personas).** Preparan a propósito
cuatro muestras de la misma base (arábica, tueste medio): una fresca, una
sobretostada, una expuesta a humedad y una rancia/con moho. Como ustedes las
prepararon, saben con certeza la categoría real de cada una. Con
`capture_webcam.py` fotografían cada muestra bajo la etiqueta correcta —
ustedes etiquetan, el modelo todavía no sabe nada.

**2. Uso (lo hace el sistema solo).** Con el modelo ya entrenado, se le
muestra cualquier grano y el sistema lo clasifica por sí mismo, sin que nadie
le diga la respuesta. `test_webcam_inference.py` simula esta etapa en la PC,
y más adelante el mismo modelo corre así dentro del ESP32-CAM.

Las 4 categorías coinciden con las definidas en el documento del proyecto:

- `calidad_alta` — muestra fresca, tueste correcto
- `quemado` — muestra sobretostada
- `humedad` — muestra expuesta a humedad
- `rancio` — muestra rancia o con moho

## 1. Capturar fotos reales con la webcam (etapa de entrenamiento)

```bash
pip install opencv-python
python capture_webcam.py --out dataset
```

Preparen físicamente una de las cuatro muestras, apúntenla con la webcam y
presionen la tecla de su categoría (`1` a `4`) varias veces —moviendo
ligeramente el grano o el ángulo entre foto y foto— para acumular varias
fotos de esa misma muestra ya sabida. Repitan para las otras tres muestras.
El script recorta automáticamente al centro y guarda en
`dataset/train/<clase>/` y, de vez en cuando, en `dataset/val/<clase>/` para
tener con qué validar.

Recomendación: mantengan el grano ocupando aproximadamente la misma posición
y tamaño en el encuadre que ocupará más adelante frente a la cámara del
ESP32-CAM, y usen una iluminación pareja — cuanto más se parezca el encuadre
de la webcam al que tendrá el dispositivo final, más útil será lo aprendido
ahora al portar el modelo.

*(Alternativa: `generate_synthetic_data.py` genera granos sintéticos falsos
solo para probar que el pipeline completo corre de punta a punta, sin
esperar a tener las muestras físicas listas. Ya no es el paso recomendado
ahora que tienen la webcam, pero sigue ahí por si quieren una prueba rápida.)*

## 2. Entrenar y exportar a TFLite

```bash
pip install tensorflow pillow numpy
python train_cnn.py --data dataset --epochs 20 --out model_out
```

Entrena una CNN pequeña y deja en `model_out/`:

- `cnn_cafe.h5` — modelo Keras completo (para seguir iterando en la PC)
- `model_float32.tflite` — para verificar que la conversión a TFLite no
  cambió el comportamiento
- `model_int8.tflite` — **la que eventualmente se integra al firmware**,
  cuantizada a 8 bits
- `class_names.txt` — el orden de las clases (debería salir
  `calidad_alta, humedad, quemado, rancio` en orden alfabético, que es como
  `image_dataset_from_directory` los toma de los nombres de carpeta)

## 3. Probar el reconocimiento en vivo con la webcam (etapa de uso)

```bash
python test_webcam_inference.py --model model_out/cnn_cafe.h5 --classes model_out/class_names.txt
```

Muestra la cámara en vivo con la clasificación que el modelo va dando por su
cuenta, más una barra de confianza por cada categoría — esta es la etapa de
uso simulada en la PC: nadie etiqueta nada aquí, el sistema decide solo.

Para verificar específicamente que la versión ya cuantizada (la que de
verdad va al ESP32-CAM) sigue clasificando bien:

```bash
python test_webcam_inference.py --model model_out/model_int8.tflite --classes model_out/class_names.txt --tflite
```

## 4. Convertir a un arreglo de C para el firmware (cuando tengan el ESP32)

```bash
python export_tflite_to_c.py model_out/model_int8.tflite --out model_out/model_data.h
```

Genera `model_data.h` con el modelo embebido como arreglo de bytes, listo
para incluirse en el proyecto de Arduino/ESP-IDF y cargarse con
TensorFlow Lite for Microcontrollers (`tflite::GetModel(g_cnn_cafe_model)`).

## Notas para cuando se integre al ESP32-CAM

- **Tamaño de imagen**: el modelo espera 96×96 RGB, igual con la webcam que
  con la OV2640 del ESP32-CAM. El firmware debe recortar y redimensionar el
  frame de la cámara a ese tamaño antes de pasarlo al modelo.
- **Diferencia de cámara**: una webcam de laptop y la OV2640 del ESP32-CAM
  no ven el color exactamente igual (lente, balance de blancos, resolución).
  Es normal que un modelo entrenado solo con fotos de webcam pierda algo de
  precisión al pasar a la cámara real; por eso, en cuanto tengan el ESP32
  armado, conviene recapturar al menos una parte del dataset directamente
  con su cámara y reentrenar antes de dar el modelo por definitivo.
- **Normalización vs. cuantización**: el modelo entrenado normaliza a
  `[0, 1]`, pero el `.tflite` int8 espera bytes ya cuantizados. El firmware
  debe usar los parámetros de cuantización que trae el propio `.tflite`
  (`input->params.scale` y `input->params.zero_point`), tal como ya hace
  `test_webcam_inference.py --tflite` en la PC.
- **Memoria**: si `model_int8.tflite` supera ~250-300 KB, conviene reducir
  aún más los filtros (`Conv2D(8/16/32,...)` en `train_cnn.py`).
