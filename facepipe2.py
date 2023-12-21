#Detector y maya facial dentro del recuadro

import cv2
import mediapipe as mp

# Inicializar el módulo de MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Definir el tamaño y la posición de la región de interés (ROI) en el centro de la ventana
roi_size = 370
roi_x = int((640 - roi_size) / 2)
roi_y = int((480 - roi_size) / 2)

while True:
    # Capturar fotograma de la cámara
    ret, frame = cap.read()

    # Obtener puntos de la malla facial con MediaPipe en la imagen en color
    results = face_mesh.process(frame)

    # Verificar si se detectaron caras y si la primera cara está dentro de la ROI
    if results.multi_face_landmarks and results.multi_face_landmarks[0]:
        landmarks = results.multi_face_landmarks[0]

        # Convertir las coordenadas normalizadas a píxeles
        h, w, _ = frame.shape
        landmarks_pixels = [(int(l.x * w), int(l.y * h)) for l in landmarks.landmark]

        # Dibujar la región de interés (ROI) en el centro de la ventana
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (255, 0, 0), 2)

        # Dibujar puntos faciales y conexiones solo si el rostro está dentro de la ROI
        for landmark_pixel in landmarks_pixels:
            x_landmark, y_landmark = landmark_pixel
            if roi_x < x_landmark < roi_x + roi_size and roi_y < y_landmark < roi_y + roi_size:
                cv2.circle(frame, (x_landmark, y_landmark), 1, (255, 250, 240), -1)

        # Definir las conexiones entre los puntos
        face_edges = mp_face_mesh.FACEMESH_CONTOURS   
        for edge in face_edges:
            x1, y1 = landmarks_pixels[edge[0]]
            x2, y2 = landmarks_pixels[edge[1]]
            cv2.line(frame, (x1, y1), (x2, y2), (255, 250, 240), 1)

    # Mostrar el fotograma con las detecciones
    cv2.imshow('Face Landmarks', frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
