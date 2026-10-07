"""
Ayuda a identificar qué índice --cam corresponde a cada cámara conectada
(la interna de la laptop vs. la webcam externa), sin tener que adivinar.

Abre cada índice del 0 al 4 uno por uno, muestra unos segundos de video en
una ventana con el número de índice en el título, y pasa al siguiente.

Uso:
    python list_cameras.py
"""
import cv2

MAX_INDEX_TO_TRY = 5
SECONDS_PER_CAMERA = 4

for i in range(MAX_INDEX_TO_TRY):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"--cam {i}: no hay ninguna cámara en este índice.")
        cap.release()
        continue

    print(f"--cam {i}: cámara encontrada. Mostrando video unos segundos... "
          f"fíjate si es la interna o la externa.")

    frames_to_show = SECONDS_PER_CAMERA * 20  # aprox., no depende del FPS real
    for _ in range(frames_to_show):
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(frame, f"--cam {i}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Identificando camaras", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

print("\nListo. Usa el número --cam que correspondía a la ventana donde viste "
      "la imagen de tu webcam externa, por ejemplo:\n"
      "  python capture_webcam.py --out dataset --cam <ese número>")