#MAYA FACIAL DE LA LIBRERÍA DLIB 68 LANDMARKS DENTRO DEL RECUADRO


import cv2
import dlib

# Cargar el detector de caras de Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("app/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Definir el tamaño y la posición de la región de interés (ROI) en el centro de la ventana
roi_size = 370
roi_x = int((640 - roi_size) / 2)
roi_y = int((480 - roi_size) / 2)

while True:
    # Capturar fotograma de la cámara
    ret, frame = cap.read()

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en el fotograma
    faces = detector(gray)

    # Dibujar la región de interés (ROI) en el centro de la ventana
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (255, 0, 0), 2)

    for face in faces:
        # Obtener puntos faciales
        landmarks = predictor(gray, face)

        # Obtener las coordenadas del rectángulo alrededor del rostro
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Verificar si el rostro está dentro de la ROI
        if x > roi_x and x + w < roi_x + roi_size and y > roi_y and y + h < roi_y + roi_size:
            # Dibujar rectángulo alrededor del rostro en el frame original
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Dibujar puntos faciales solo si el rostro está dentro de la ROI
            for i in range(68):
                x_landmark, y_landmark = landmarks.part(i).x, landmarks.part(i).y
                cv2.circle(frame, (x_landmark, y_landmark), 2, (0, 0, 255), -1)

    # Mostrar el fotograma con las detecciones
    cv2.imshow('Face Landmarks', frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
