import cv2
import numpy as np
import matplotlib.pyplot as plt

class Preprocessing:
    
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = None
        self.magnitude_spectrum = None
        
    def preprocess_image(self, size=(256, 256)):
        img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        img = img / 255.0
        self.img = img
        return img

    def fourier_transform(self):
        if self.img is None:
            raise ValueError("Use preprocess_image() antes de chamar fourier_transform().")
        
        f = np.fft.fft2(self.img)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        self.magnitude_spectrum = magnitude
        return magnitude

    def show(self):
        if self.img is None or self.magnitude_spectrum is None:
            raise ValueError("Chame preprocess_image() e fourier_transform() antes de show().")

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(self.img, cmap='gray')
        plt.title('Imagem Processada')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(self.magnitude_spectrum, cmap='gray')
        plt.title('Transformada de Fourier')
        plt.axis('off')

        plt.tight_layout()
        plt.show()