import cv2
import numpy as np
from tensorflow import keras

# Configurações
MODELO_PATH = "modelo/modelo_mascara.h5"
IMG_TAMANHO = (100, 100)
CONFIANCA_MINIMA = 0.6

# Cores (BGR)
VERDE = (0, 255, 0)
VERMELHO = (0, 0, 255)
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)

def carregar_modelo():
    print("📂 A carregar modelo...")
    modelo = keras.models.load_model(MODELO_PATH)
    print("✅ Modelo carregado!\n")
    return modelo


def pre_processar_rosto(rosto):
    rosto = cv2.resize(rosto, IMG_TAMANHO)
    rosto = cv2.cvtColor(rosto, cv2.COLOR_BGR2RGB)
    rosto = rosto / 255.0
    rosto = np.expand_dims(rosto, axis=0)
    return rosto


def desenhar_caixa(frame, x, y, w, h, label, confianca, cor):
    # Caixa ao redor do rosto
    cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)

    # Fundo do texto
    texto = f"{label} ({confianca:.0f}%)"
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), cor, -1)

    # Texto
    cv2.putText(
        frame, texto,
        (x + 5, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, BRANCO, 2
    )


def iniciar_detector():
    modelo = carregar_modelo()

    # Detector de rostos
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Abrir câmara
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Erro: Câmara não encontrada!")
        return

    print("🎥 Câmara iniciada! Prima 'Q' para sair.\n")

    # Contadores
    com_mascara = 0
    sem_mascara = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Espelhar imagem
        frame = cv2.flip(frame, 1)
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostos
        rostos = face_cascade.detectMultiScale(
            cinza,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        com_mascara = 0
        sem_mascara = 0

        for (x, y, w, h) in rostos:
            rosto = frame[y:y + h, x:x + w]
            rosto_processado = pre_processar_rosto(rosto)

            # Previsão
            prob = modelo.predict(rosto_processado, verbose=0)[0][0]
            confianca = prob * 100 if prob > 0.5 else (1 - prob) * 100

            if prob > 0.5:
                label = "Com Mascara"
                cor = VERDE
                com_mascara += 1
            else:
                label = "Sem Mascara"
                cor = VERMELHO
                sem_mascara += 1

            desenhar_caixa(frame, x, y, w, h, label, confianca, cor)

        # Painel de informação no topo
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), PRETO, -1)
        cv2.putText(frame, f"Com Mascara: {com_mascara}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, VERDE, 2)
        cv2.putText(frame, f"Sem Mascara: {sem_mascara}",
                    (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, VERMELHO, 2)
        cv2.putText(frame, "Prima Q para sair",
                    (frame.shape[1] - 220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BRANCO, 1)

        # Mostrar frame
        cv2.imshow("Detector de Mascara - Deep Learning", frame)

        # Sair com Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Detector encerrado!")


if __name__ == "__main__":
    iniciar_detector()