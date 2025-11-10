import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função de pré-processamento
def preprocess_image(img_path, size=(256, 256)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img = img / 255.0  # normalização
    return img

# Função para aplicar a Transformada de Fourier
def fourier_transform(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

# Caminho da imagem (altere aqui)
img_path = "TexVision/dataset/Fabric Defect Dataset/defect free/809af08c6a3824711208578173.jpg"

# Pré-processar e aplicar Fourier
img = preprocess_image(img_path)
magnitude_spectrum = fourier_transform(img)

# Exibir os resultados
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Transformada de Fourier (Magnitude)')
plt.axis('off')

plt.tight_layout()
plt.show()