#EXTRACTOR DE DESCRIPTORES FACIALES (ni idea como lo hace)


import face_recognition


# Cargar la imagen a analizar
image = face_recognition.load_image_file("app/face/referencia.jpeg")
# Detectar rostros en la imagen
faces = face_recognition.face_locations(image)
# Procesar cada rostro detectado
for face in faces:
    # Extraer las características faciales del rostro detectado
    facial_features1 = face_recognition.face_encodings(image, [face])

print (facial_features1)



#Extrae caracaterísticas faciales pero no se en que formato xd