from preprocessing import *
from load_Image import *
from model import AutoEncoder
from fourier import Fourier
from datetime import datetime

# folder = r"TexVision\\Images\\"
folder = r"TexVision\\IA\\img_valid"
model = "TexVision\\IA\\checkpoints\\modelo.keras"
img_OK = "TexVision\\images_OK"
img_BAD = "TexVision\\images_BAD"
list_imgs = carregar_imagens(folder)
print("Inicio: " + str(datetime.now()))

X = []

for img_path in list_imgs:
    pre_processing = Preprocessing(img_path)
    pre_processing.remove_errors( img_OK, img_BAD)

folder = r"TexVision\\Images_OK"
list_imgs_OK = carregar_imagens(folder)

for img_path in list_imgs_OK:

    pre_processing = Preprocessing(img_path)
    pre_processing = pre_processing.preprocess_image()

    fourier = Fourier(pre_processing)
    fourier = fourier.fourier_transform()
    X.append(fourier)

    value = AutoEncoder(fourier, img_path)
    value.play(model, img_OK, img_BAD)


# X = np.array(X, dtype=np.float32)
# print("Dataset Fourier shape:", X.shape)
# value = AutoEncoder(X)
# value.create_model(model)
# print(X)
print("Fim: " + str(datetime.now()))

###################################################################



# for img_path in list_imgs_OK:

#     pre_processing = Preprocessing(img_path)
#     pre_processing = pre_processing.preprocess_image()

#     fourier = Fourier(pre_processing)
#     fourier = fourier.fourier_transform()

#     ft = fourier

#     X.append(ft)
# value = AutoEncoder(X)
# value.create_model()
# value.play(model, img_OK, img_BAD)