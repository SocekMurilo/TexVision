import numpy as np
import os
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class AutoEncoder:

    def __init__(self, x, img_path):
        self.x = x.reshape(1, 256, 256, 1)
        self.img_path = img_path
        self.threshold = None

    def build_autoencoder(input_shape=(256, 256, 1)):
        inputs = layers.Input(shape=input_shape)

        # ENCODER
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D(2, padding='same')(x)

        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2, padding='same')(x)

        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D(2, padding='same')(x)

        # DECODER
        x = layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(encoded)
        x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)

        outputs = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')

        return model

    def create_model(self, model):
        X = self.x

        if X is None or len(X) == 0:
            # raise ValueError("Não há imagens para serem processadas")
            print("⚠️ Treinamento ignorado: dataset vazio.")
            return None
            
        autoencoder = AutoEncoder.build_autoencoder()
        autoencoder.summary()

        history = autoencoder.fit(
            X, X,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            shuffle=True
        )

        autoencoder.save(model)

    def play( self, model_path, img_OK, img_BAD):

        autoencoder = load_model(model_path)

        os.makedirs(img_OK, exist_ok=True)
        os.makedirs(img_BAD, exist_ok=True)

        # Usa o threshold já salvo/calculado
        threshold = self.threshold
        print(f"📌 Usando threshold = {threshold}")

        recon = autoencoder.predict(self.x)

        erro = np.mean((self.x - recon)**2)
        median = np.median(erro)
        mad = np.median(np.abs(erro - median))
        threshold = median + 3 * mad
        self.threshold = 0.003#

        print(erro)
        print(threshold)

        if erro < threshold:
            destino = img_OK
        else:
            destino = img_BAD

        shutil.copy(self.img_path, destino)

        print(f"{os.path.basename(self.img_path)} -> Erro {erro:.6f} -> {'BOA' if erro < threshold else 'RUIM'}")

        print("\n🔍 Classificação concluída!")
        print(f"✔ Imagens boas: {img_OK}")
        print(f"✔ Imagens ruins: {img_BAD}")


# erro = lista de erros já calculados
# threshold = seu threshold já calculado
# y_true = labels verdadeiros (0 = bom, 1 = defeito)

        # y_pred = (erro > threshold).astype(int)

        # print("Acurácia:", accuracy_score(y_true, y_pred))
        # print(confusion_matrix(y_true, y_pred))
        # print(classification_report(y_true, y_pred))