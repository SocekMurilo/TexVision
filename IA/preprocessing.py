from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from skimage.color import deltaE_ciede2000
from data_failed import *


class Preprocessing:
    
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = None
        
    def preprocess_image(self, size=(256, 256)):
        img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Erro ao ler a imagem: {self.img_path}")

        img = self.resize_with_padding(img, size)

        if img.shape != size:
            raise ValueError(f"Imagem não ficou 256x256: {self.img_path}, shape:{img.shape}")

        img = img / 255.0
        self.img = img
        return self.img_path

    def resize_with_padding(self, img, size): # serve para ele quando ser aplicado o resize manter a proporção
        h, w = img.shape
        target_h, target_w = size

        scale = min(target_h / h, target_w / w)
        nh, nw = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        top = (target_h - nh) // 2
        bottom = target_h - nh - top
        left = (target_w - nw) // 2
        right = target_w - nw - left

        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=0
        )

        return img_padded

    def remove_errors(self, img_valid, img_bad, limite_delta=20, sample_rate=0.05):

        # === Lê imagem colorida ===
        img_color = cv2.imread(self.img_path)
        if img_color is None:
            raise ValueError(f"Erro ao ler a imagem: {self.img_path}")

        lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
        h, w, _ = lab.shape
        total_pixels = h * w
        n_samples = int(total_pixels * sample_rate)
        idx = np.random.choice(total_pixels, n_samples, replace=False)
        sampled_lab = lab.reshape(-1, 3)[idx]
        media_lab = np.mean(sampled_lab, axis=0, keepdims=True)
        delta = deltaE_ciede2000(sampled_lab, media_lab)
        delta95 = np.percentile(delta, 95)

        if delta95 > limite_delta:

            end_preocces = str(datetime.now())
            print("Horario da Falha: " + end_preocces)

            if not os.path.exists(img_bad):
                os.makedirs(img_bad)

            new_path = os.path.join(img_bad, os.path.basename(self.img_path))
            shutil.move(self.img_path, new_path)

            self.img_path = new_path
            return new_path

            # if os.path.exists(self.img_path):
            #     os.remove(self.img_path)
            #     return True
            # else:
            #     raise FileNotFoundError(f"Arquivo não encontrado: {self.img_path}")
        

        # # === Move para pasta válida ===
        if not os.path.exists(img_valid):
            os.makedirs(img_valid)

        new_path = os.path.join(img_valid, os.path.basename(self.img_path))
        shutil.move(self.img_path, new_path)

        self.img_path = new_path
        return new_path
