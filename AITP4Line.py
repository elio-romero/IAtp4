import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carga la imagen
image_path = 'route.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detección de bordes usando Canny
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Transformada de Hough para deteccion de linea
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Crea una copia de la imagen original para dibujar líneas.
line_image = np.copy(image)

# Dibuja las linesa sobre la copia de la imagen original
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Muestra el resultado
plt.figure(figsize=(10, 10))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
plt.subplot(122), plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)), plt.title('Lineas Detectadas ')
plt.show()
