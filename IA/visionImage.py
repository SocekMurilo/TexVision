import kagglehub
from preprocess import *

# # Download latest version
# path = kagglehub.dataset_download("nexuswho/fabric-defects-dataset")

# print("Path to dataset files:", path)

image = "TexVision\\dataset\\Fabric Defect Dataset\\defect free\\0000000.jpg"

new_image = Preprocessing(image)

new_image.preprocess_image()
new_image.fourier_transform()
new_image.show()