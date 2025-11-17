import os
import numpy as np
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import image

# =============================================
# CONFIGURAÇÕES
# =============================================
path_img = "TexVision/dataset/Fabric Defect Dataset/teste/"
model_path = "checkpoints/autoencoder.keras"
batch_size = 32
epochs = 50
image_size = (128, 128)

# =============================================
# CARREGANDO DATASET (APENAS IMAGENS BOAS)
# =============================================
train = image_dataset_from_directory(
    path_img,
    labels=None,            # NÃO usa classes
    seed=123,
    shuffle=True,
    image_size=image_size,
    batch_size=batch_size
)

# =============================================
# MODELO AUTOENCODER
# =============================================

# ENCODER
encoder = models.Sequential([
    layers.Input(shape=(128,128,3)),
    layers.Rescaling(1/255),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
])

# DECODER
decoder = models.Sequential([
    layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same'),
    layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same'),
    layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same'),
    layers.Conv2D(3, 3, activation='sigmoid', padding='same')  # imagem normalizada
])

# AUTOENCODER COMPLETO
autoencoder = models.Sequential([encoder, decoder])

autoencoder.compile(
    optimizer=optimizers.Adam(1e-3),
    loss='mse'
)

autoencoder.summary()

# =============================================
# TREINO
# =============================================

autoencoder.fit(
    train,
    epochs=epochs
)

# =============================================
# SALVANDO MODELO
# =============================================
os.makedirs("checkpoints", exist_ok=True)
autoencoder.save(model_path)

print("Modelo salvo em:", model_path)

# =============================================
# FUNÇÃO DE DETECÇÃO DE DEFEITOS
# =============================================

def detectar_defeito(model, img_path, threshold=0.02):
    """
    Retorna True se detectar defeito.
    False se imagem for normal.
    """

    img = image.load_img(img_path, target_size=(128,128))
    x = image.img_to_array(img) / 255.
    x = np.expand_dims(x, axis=0)

    reconstruida = model.predict(x)
    erro = np.mean(np.abs(x - reconstruida))

    print("Erro da imagem:", erro)

    return erro > threshold
