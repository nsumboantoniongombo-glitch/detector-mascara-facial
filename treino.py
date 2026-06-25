import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocessamento import carregar_dados, dividir_dados

# Configurações
IMG_TAMANHO = (100, 100)
EPOCHS = 10
BATCH_SIZE = 32
MODELO_PATH = "modelo/modelo_mascara.h5"

def criar_modelo():
    # Usar MobileNetV2 como base (Transfer Learning)
    base_modelo = MobileNetV2(
        input_shape=(100, 100, 3),
        include_top=False,
        weights="imagenet"
    )
    base_modelo.trainable = False  # congelar pesos base

    modelo = keras.Sequential([
        base_modelo,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")  # 0 ou 1
    ])

    modelo.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("✅ Modelo criado com sucesso!")
    modelo.summary()
    return modelo


def aumentar_dados(X_treino, y_treino):
    # Aumentar dados para melhorar o treino
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2
    )
    datagen.fit(X_treino)
    return datagen


def treinar_modelo():
    # Carregar e dividir dados
    print("📂 A carregar dados...\n")
    imagens, etiquetas = carregar_dados()
    X_treino, X_teste, y_treino, y_teste = dividir_dados(imagens, etiquetas)

    # Criar modelo
    print("\n🧠 A criar modelo...\n")
    modelo = criar_modelo()

    # Aumentar dados
    datagen = aumentar_dados(X_treino, y_treino)

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            MODELO_PATH,
            save_best_only=True
        )
    ]

    # Treinar
    print("\n🚀 A iniciar treino...\n")
    historico = modelo.fit(
        datagen.flow(X_treino, y_treino, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_teste, y_teste),
        callbacks=callbacks
    )

    # Avaliar
    loss, accuracy = modelo.evaluate(X_teste, y_teste)
    print(f"\n✅ Treino concluído!")
    print(f"   📊 Precisão no teste: {accuracy * 100:.2f}%")
    print(f"   💾 Modelo guardado em: {MODELO_PATH}")

    # Gráfico
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(historico.history["accuracy"], label="Treino")
    plt.plot(historico.history["val_accuracy"], label="Validação")
    plt.title("Precisão do Modelo")
    plt.xlabel("Época")
    plt.ylabel("Precisão")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(historico.history["loss"], label="Treino")
    plt.plot(historico.history["val_loss"], label="Validação")
    plt.title("Perda do Modelo")
    plt.xlabel("Época")
    plt.ylabel("Perda")
    plt.legend()

    plt.tight_layout()
    plt.savefig("assets/grafico_treino.png")
    plt.show()
    print("📈 Gráfico guardado em assets/grafico_treino.png")


if __name__ == "__main__":
    treinar_modelo()