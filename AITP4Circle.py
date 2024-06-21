import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
image_path = 'aro.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Desenfoca la imagen para reducir el ruido.
gray_blurred = cv2.medianBlur(gray, 5)

# Transformada de Hough para detección de círculos
circles = cv2.HoughCircles(
    gray_blurred, 
    cv2.HOUGH_GRADIENT, 
    dp=1, 
    minDist=100,
    param1=100, 
    param2=90, 
    minRadius=100, 
    maxRadius=1000
)

# Crea una copia de la imagen original para dibujar círculos.
circle_image = np.copy(image)

# Dibuja los círculos detectados
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Dibuja el círculo exterior
        cv2.circle(circle_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Dibuja el centro del círculo
        cv2.circle(circle_image, (i[0], i[1]), 2, (0, 0, 255), 3)

# Muestra el resultado
plt.figure(figsize=(10, 10))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
plt.subplot(122), plt.imshow(cv2.cvtColor(circle_image, cv2.COLOR_BGR2RGB)), plt.title(' Circulos Detectados')
plt.show()
