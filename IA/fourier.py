
import cv2
import numpy as np

class Fourier:

    def __init__(self, img_path):
        self.img_path = img_path
        self.img = None
        self.magnitude_spectrum = None

    def fourier_transform(self, highpass=False, lowpass=False, cutoff=10, target_size=(256, 256)):
        """
        Calcula o espectro de Fourier da imagem, garantindo saída 2D completa
        no formato (H, W, 1) compatível com Autoencoder.
        """

        # --- 1) Carrega imagem diretamente, sem precisar de preprocess() ---
        if self.img_path is not None:
            img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Não foi possível abrir a imagem: {self.img_path}")
            img = cv2.resize(img, target_size)
            img = img.astype(np.float32) / 255.0
        else:
            if self.img is None:
                raise ValueError("Nenhuma imagem carregada. Use preprocess_image() ou img_path.")
            img = self.img.astype(np.float32)

        # --- 2) Aplica janela Hanning para suavizar bordas ---
        h, w = img.shape
        window = np.hanning(h)[:, None] * np.hanning(w)[None, :]
        img_windowed = img * window

        # --- 3) FFT COMPLETA (2D) + shift para centralizar frequências ---
        F = np.fft.fft2(img_windowed)
        F_shift = np.fft.fftshift(F)

        # --- 4) Filtros opcionais (HPF/LPF) ---
        if highpass or lowpass:
            Y, X = np.ogrid[:h, :w]
            cy, cx = h // 2, w // 2
            dist = np.sqrt((Y - cy)**2 + (X - cx)**2)

            if highpass:
                F_shift[dist < cutoff] = 0
            if lowpass:
                F_shift[dist > cutoff] = 0

        # --- 5) Espectro de magnitude ---
        magnitude = np.abs(F_shift)
        magnitude = np.log1p(magnitude)  # log para compressão

        # --- 6) Normalização 0-1 ---
        magnitude -= magnitude.min()
        magnitude /= (magnitude.max() + 1e-8)

        # --- 7) Garante formato (H, W, 1) ---
        magnitude = magnitude.astype(np.float32)
        magnitude = np.expand_dims(magnitude, axis=-1)

        # guarda internamente
        self.magnitude_spectrum = magnitude

        return magnitude
    
    # def show(self):
    #     if self.img is None or self.magnitude_spectrum is None:
    #         raise ValueError("Chame preprocess_image() e fourier_transform() antes de show().")

    #     plt.figure(figsize=(10, 5))

    #     plt.subplot(1, 2, 1)
    #     plt.imshow(self.img, cmap='gray')
    #     plt.title('Imagem Processada')
    #     plt.axis('off')

    #     plt.subplot(1, 2, 2)
    #     plt.imshow(self.magnitude_spectrum, cmap='gray')
    #     plt.title('Transformada de Fourier')
    #     plt.axis('off')

    #     plt.tight_layout()
    #     plt.show()