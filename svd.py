from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np 
import os

# Leer la imagen y cambiar el tamaño y color
plt.rcParams['figure.figsize'] = [16, 8]

# Leyendo el archivo y haciendo a escala de grises
image_path = imread('gatito.jpeg')
X = np.mean(image_path, -1)
img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')

# Generando SVD
U, Sigma, VT = np.linalg.svd(X, full_matrices = False)

plt.show()
# Extraer SVD de la matrix Sigma
Sigma = np.diag(Sigma)

i = 0
for k in (5, 20, 100):
    # @ es para multiplicar matrices
    X_Ap = U[:,:k] @ Sigma[0:k,:k] @ VT[:k,:]
    plt.figure(i+1)
    i += 1
    img = plt.imshow(X_Ap)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('k = ' + str(k))
    plt.show()