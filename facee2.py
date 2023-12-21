#Este detecta el rostro y dibuja los puntos, (FALLOS AL DETECTAR ROSTRO)
import cv2
import dlib

# Cargar el detector de caras de Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("app/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")  # Descargar desde http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar fotograma de la cámara
    ret, frame = cap.read()

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en el fotograma
    faces = detector(gray)

    for face in faces:
        # Obtener puntos faciales
        landmarks = predictor(gray, face)

        # Dibujar rectángulo alrededor del rostro
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Dibujar puntos faciales
        for i in range(68):  # Dlib proporciona 68 puntos faciales
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            
            
            

    # Mostrar el fotograma con las detecciones
    cv2.imshow('Face Landmarks', frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()