import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

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
    
    def remove_errors(self, limite_delta=16, img_valid="img_valid"):
        if self.img is None:
            raise ValueError("Precisa preprocessar a imagem antes de validar.")

        img_color = cv2.imread(self.img_path)
        if img_color is None:
            raise ValueError(f"Erro ao ler a imagem: {self.img_path}")

        lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3)
        media = np.mean(pixels, axis=0)
        delta = np.linalg.norm(pixels - media, axis=1)
        media_delta = np.mean(delta)
        print(f"Desvio médio de cor (DeltaE): {media_delta:.2f}")
        if media_delta > limite_delta:
            print("Imagem rejeitada (cor fora do padrão).")
            return False
        if not os.path.exists(img_valid):
            os.makedirs(img_valid)

        new_path = os.path.join(img_valid, os.path.basename(self.img_path))

        shutil.move(self.img_path, new_path)

        print(f"Imagem válida. Movida para: {new_path}")
        return True
        
        
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