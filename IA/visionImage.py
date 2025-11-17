import kagglehub
from pathlib import Path
from preprocess import *
import cv2
import os 

# # Download latest version
# path = kagglehub.dataset_download("nexuswho/fabric-defects-dataset")

# print("Path to dataset files:", path)

path = '"TexVision\\dataset\\Fabric Defect Dataset\\teste\\"'

def carregar_imagens(folder):
    list_paths = []

    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            list_paths.append(os.path.join(folder, file))

    return list_paths

folder = r"TexVision\\dataset\\Fabric Defect Dataset\\teste\\"
list_imgs = carregar_imagens(folder)

print(f"{list_imgs[0]} imagens carregadas com sucesso.")

# image = "TexVision\\dataset\\Fabric Defect Dataset\\defect free\\0000000.jpg"

for imgs in range(len(list_imgs)):
    new_image = Preprocessing(list_imgs[imgs])

    new_image.preprocess_image()
    new_image.remove_errors()
    new_image.fourier_transform()
    # new_image.show()