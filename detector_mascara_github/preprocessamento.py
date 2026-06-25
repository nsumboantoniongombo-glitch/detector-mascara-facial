import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Configurações
IMG_TAMANHO = (100, 100)  # tamanho padrão de cada imagem
DATASET_PATH = "dataset"

def carregar_dados():
    imagens = []
    etiquetas = []

    categorias = {"com_mascara": 1, "sem_mascara": 0}

    for categoria, etiqueta in categorias.items():
        caminho = os.path.join(DATASET_PATH, categoria)
        
        if not os.path.exists(caminho):
            print(f"⚠️  Pasta não encontrada: {caminho}")
            continue

        ficheiros = os.listdir(caminho)
        print(f"📂 {categoria}: {len(ficheiros)} imagens encontradas")

        for ficheiro in ficheiros:
            if ficheiro.lower().endswith((".jpg", ".jpeg", ".png")):
                caminho_img = os.path.join(caminho, ficheiro)
                try:
                    img = Image.open(caminho_img).convert("RGB")
                    img = img.resize(IMG_TAMANHO)
                    img_array = np.array(img) / 255.0  # normalizar 0-1
                    imagens.append(img_array)
                    etiquetas.append(etiqueta)
                except Exception as e:
                    print(f"❌ Erro ao carregar {ficheiro}: {e}")

    imagens = np.array(imagens)
    etiquetas = np.array(etiquetas)

    print(f"\n✅ Total de imagens carregadas: {len(imagens)}")
    print(f"   Com máscara: {sum(etiquetas == 1)}")
    print(f"   Sem máscara: {sum(etiquetas == 0)}")

    return imagens, etiquetas


def dividir_dados(imagens, etiquetas):
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        imagens, etiquetas,
        test_size=0.2,       # 20% para teste
        random_state=42,
        stratify=etiquetas
    )

    print(f"\n📊 Divisão dos dados:")
    print(f"   Treino: {len(X_treino)} imagens")
    print(f"   Teste:  {len(X_teste)} imagens")

    return X_treino, X_teste, y_treino, y_teste


# Teste rápido
if __name__ == "__main__":
    imagens, etiquetas = carregar_dados()
    X_treino, X_teste, y_treino, y_teste = dividir_dados(imagens, etiquetas)