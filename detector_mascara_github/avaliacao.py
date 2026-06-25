import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from preprocessamento import carregar_dados, dividir_dados

# Configurações
MODELO_PATH = "modelo/modelo_mascara.h5"
CLASSES = ["Sem Máscara", "Com Máscara"]


def carregar_modelo():
    print("📂 A carregar modelo treinado...")
    modelo = keras.models.load_model(MODELO_PATH)
    print("✅ Modelo carregado com sucesso!\n")
    return modelo


def avaliar_modelo():
    # Carregar dados e modelo
    imagens, etiquetas = carregar_dados()
    _, X_teste, _, y_teste = dividir_dados(imagens, etiquetas)
    modelo = carregar_modelo()

    # Fazer previsões
    print("🔍 A fazer previsões...\n")
    previsoes_prob = modelo.predict(X_teste)
    previsoes = (previsoes_prob > 0.5).astype(int).flatten()

    # Relatório completo
    print("📊 Relatório de Classificação:")
    print("-" * 40)
    print(classification_report(y_teste, previsoes, target_names=CLASSES))

    # Matriz de confusão
    cm = confusion_matrix(y_teste, previsoes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusão - Detector de Máscara")
    plt.tight_layout()
    plt.savefig("assets/matriz_confusao.png")
    plt.show()
    print("📈 Matriz guardada em assets/matriz_confusao.png")

    # Mostrar algumas previsões
    mostrar_previsoes(X_teste, y_teste, previsoes)


def mostrar_previsoes(X_teste, y_teste, previsoes, n=10):
    plt.figure(figsize=(15, 5))
    indices = np.random.choice(len(X_teste), n, replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_teste[idx])
        plt.axis("off")

        real = CLASSES[y_teste[idx]]
        previsto = CLASSES[previsoes[idx]]
        cor = "green" if real == previsto else "red"

        plt.title(f"Real: {real}\nPrevisto: {previsto}", color=cor, fontsize=8)

    plt.suptitle("Exemplos de Previsões", fontsize=14)
    plt.tight_layout()
    plt.savefig("assets/exemplos_previsoes.png")
    plt.show()
    print("🖼️  Exemplos guardados em assets/exemplos_previsoes.png")


if __name__ == "__main__":
    avaliar_modelo()