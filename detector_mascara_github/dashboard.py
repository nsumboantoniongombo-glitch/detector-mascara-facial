import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from PIL import Image
import time
import os

# Configurações
MODELO_PATH = "modelo/modelo_mascara.h5"
IMG_TAMANHO = (100, 100)

# Configuração da página
st.set_page_config(
    page_title="Detector de Máscara",
    page_icon="😷",
    layout="wide"
)

# Estilo CSS
st.markdown("""
    <style>
        .titulo {
            text-align: center;
            color: #2ecc71;
            font-size: 2.5em;
            font-weight: bold;
        }
        .subtitulo {
            text-align: center;
            color: #95a5a6;
            font-size: 1.1em;
        }
        .card-verde {
            background-color: #1e8449;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-size: 1.5em;
            font-weight: bold;
        }
        .card-vermelho {
            background-color: #922b21;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-size: 1.5em;
            font-weight: bold;
        }
        .card-azul {
            background-color: #1a5276;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-size: 1.5em;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)


# Título
st.markdown('<p class="titulo">😷 Detector de Máscara Facial</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitulo">Sistema de Detecção em Tempo Real usando Deep Learning</p>', unsafe_allow_html=True)
st.markdown("---")


@st.cache_resource
def carregar_modelo():
    if not os.path.exists(MODELO_PATH):
        return None
    return keras.models.load_model(MODELO_PATH)


def pre_processar(imagem):
    imagem = imagem.resize(IMG_TAMANHO)
    imagem = np.array(imagem.convert("RGB")) / 255.0
    return np.expand_dims(imagem, axis=0)


def prever(modelo, imagem):
    processada = pre_processar(imagem)
    prob = modelo.predict(processada, verbose=0)[0][0]
    if prob > 0.5:
        return "✅ Com Máscara", prob * 100, "verde"
    else:
        return "❌ Sem Máscara", (1 - prob) * 100, "vermelho"


# Sidebar
st.sidebar.image("https://img.icons8.com/emoji/96/mask-emoji.png", width=80)
st.sidebar.title("⚙️ Configurações")
modo = st.sidebar.radio(
    "Modo de Detecção:",
    ["📸 Carregar Imagem", "🎥 Câmara em Tempo Real"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Sobre o Projecto")
st.sidebar.info(
    "Modelo treinado com **Deep Learning**\n\n"
    "Arquitectura: **MobileNetV2**\n\n"
    "Técnica: **Transfer Learning**"
)

# Carregar modelo
modelo = carregar_modelo()

if modelo is None:
    st.warning("⚠️ Modelo ainda não treinado! Execute primeiro o `treino.py`")
    st.code("python treino.py", language="bash")
    st.stop()

st.success("✅ Modelo carregado com sucesso!")

# ─── MODO 1: Carregar Imagem ───────────────────────────────────────────
if modo == "📸 Carregar Imagem":
    st.subheader("📸 Detecção por Imagem")

    ficheiro = st.file_uploader(
        "Carrega uma imagem",
        type=["jpg", "jpeg", "png"]
    )

    if ficheiro:
        imagem = Image.open(ficheiro)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(imagem, caption="Imagem Carregada", use_column_width=True)

        with st.spinner("🔍 A analisar..."):
            time.sleep(0.5)
            resultado, confianca, cor = prever(modelo, imagem)

        with col2:
            st.markdown("### 🧠 Resultado")
            if cor == "verde":
                st.markdown(f'<div class="card-verde">{resultado}<br>{confianca:.1f}% confiança</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="card-vermelho">{resultado}<br>{confianca:.1f}% confiança</div>', unsafe_allow_html=True)

        with col3:
            st.markdown("### 📊 Probabilidades")
            st.metric("Com Máscara", f"{confianca:.1f}%" if cor == "verde" else f"{100 - confianca:.1f}%")
            st.metric("Sem Máscara", f"{confianca:.1f}%" if cor == "vermelho" else f"{100 - confianca:.1f}%")

        # Histórico
        if "historico" not in st.session_state:
            st.session_state.historico = []

        st.session_state.historico.append({
            "resultado": resultado,
            "confianca": f"{confianca:.1f}%"
        })

        st.markdown("---")
        st.subheader("📋 Histórico de Detecções")
        for i, item in enumerate(reversed(st.session_state.historico[-5:])):
            st.write(f"**#{i+1}** → {item['resultado']} | Confiança: {item['confianca']}")

# ─── MODO 2: Câmara em Tempo Real ─────────────────────────────────────
elif modo == "🎥 Câmara em Tempo Real":
    st.subheader("🎥 Detecção em Tempo Real")
    st.info("💡 Para detecção completa em tempo real execute: `python detector.py`")

    col1, col2 = st.columns(2)

    with col1:
        iniciar = st.button("▶️ Iniciar Câmara", type="primary")
        parar = st.button("⏹️ Parar Câmara")

    frame_placeholder = st.empty()
    resultado_placeholder = st.empty()

    col_a, col_b, col_c = st.columns(3)
    total = col_a.empty()
    com = col_b.empty()
    sem = col_c.empty()

    if iniciar:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Câmara não encontrada!")
        else:
            contador_total = 0
            while True:
                ret, frame = cap.read()
                if not ret or parar:
                    break

                frame = cv2.flip(frame, 1)
                cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rostos = face_cascade.detectMultiScale(cinza, 1.1, 5, minSize=(60, 60))

                com_mascara = 0
                sem_mascara = 0

                for (x, y, w, h) in rostos:
                    rosto_img = Image.fromarray(
                        cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                    )
                    resultado, confianca, cor = prever(modelo, rosto_img)
                    cor_box = (0, 255, 0) if cor == "verde" else (0, 0, 255)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), cor_box, 2)
                    cv2.putText(frame, f"{resultado} {confianca:.0f}%",
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                cor_box, 2)

                    if cor == "verde":
                        com_mascara += 1
                    else:
                        sem_mascara += 1

                contador_total += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                total.metric("🔢 Frames", contador_total)
                com.metric("✅ Com Máscara", com_mascara)
                sem.metric("❌ Sem Máscara", sem_mascara)

            cap.release()
            st.success("✅ Câmara encerrada!")