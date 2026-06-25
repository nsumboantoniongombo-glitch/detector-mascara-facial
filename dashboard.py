import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import time
import os

MODELO_PATH = os.path.join(os.path.dirname(__file__), "modelo", "modelo_mascara.onnx")
IMG_TAMANHO = (100, 100)

st.set_page_config(page_title="Detector de Máscara", page_icon="😷", layout="wide")

st.markdown("""
    <style>
        .titulo { text-align: center; color: #2ecc71; font-size: 2.5em; font-weight: bold; }
        .subtitulo { text-align: center; color: #95a5a6; font-size: 1.1em; }
        .card-verde { background-color: #1e8449; padding: 20px; border-radius: 10px;
            text-align: center; color: white; font-size: 1.5em; font-weight: bold; }
        .card-vermelho { background-color: #922b21; padding: 20px; border-radius: 10px;
            text-align: center; color: white; font-size: 1.5em; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="titulo">😷 Detector de Máscara Facial</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitulo">Sistema de Detecção usando Deep Learning (MobileNetV2 + ONNX)</p>', unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def carregar_modelo():
    if not os.path.exists(MODELO_PATH):
        return None
    return ort.InferenceSession(MODELO_PATH)

def pre_processar(imagem):
    imagem = imagem.resize(IMG_TAMANHO)
    imagem = np.array(imagem.convert("RGB")) / 255.0
    return np.expand_dims(imagem, axis=0).astype(np.float32)

def prever(sessao, imagem):
    entrada = pre_processar(imagem)
    nome_entrada = sessao.get_inputs()[0].name
    resultado = sessao.run(None, {nome_entrada: entrada})
    prob = resultado[0][0][0]
    if prob > 0.5:
        return "✅ Com Máscara", prob * 100, "verde"
    else:
        return "❌ Sem Máscara", (1 - prob) * 100, "vermelho"

# Sidebar
st.sidebar.image("https://img.icons8.com/emoji/96/mask-emoji.png", width=80)
st.sidebar.title("⚙️ Configurações")
modo = st.sidebar.radio("Modo de Detecção:", ["📸 Carregar Imagem"])
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Sobre o Projecto")
st.sidebar.info(
    "Modelo treinado com **Deep Learning**\n\n"
    "Arquitectura: **MobileNetV2**\n\n"
    "Técnica: **Transfer Learning**\n\n"
    "Runtime: **ONNX**"
)

sessao = carregar_modelo()

if sessao is None:
    st.warning("⚠️ Modelo não encontrado!")
    st.stop()

st.success("✅ Modelo carregado com sucesso!")

if modo == "📸 Carregar Imagem":
    st.subheader("📸 Detecção por Imagem")
    ficheiro = st.file_uploader("Carrega uma imagem", type=["jpg", "jpeg", "png"])

    if ficheiro:
        imagem = Image.open(ficheiro)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(imagem, caption="Imagem Carregada", use_column_width=True)

        with st.spinner("🔍 A analisar..."):
            time.sleep(0.3)
            resultado, confianca, cor = prever(sessao, imagem)

        with col2:
            st.markdown("### 🧠 Resultado")
            if cor == "verde":
                st.markdown(f'<div class="card-verde">{resultado}<br>{confianca:.1f}% confiança</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="card-vermelho">{resultado}<br>{confianca:.1f}% confiança</div>', unsafe_allow_html=True)

        with col3:
            st.markdown("### 📊 Probabilidades")
            prob_com = confianca if cor == "verde" else 100 - confianca
            prob_sem = 100 - prob_com
            st.metric("✅ Com Máscara", f"{prob_com:.1f}%")
            st.metric("❌ Sem Máscara", f"{prob_sem:.1f}%")

        if "historico" not in st.session_state:
            st.session_state.historico = []
        st.session_state.historico.append({"resultado": resultado, "confianca": f"{confianca:.1f}%"})

        st.markdown("---")
        st.subheader("📋 Histórico de Detecções")
        for i, item in enumerate(reversed(st.session_state.historico[-5:])):
            st.write(f"**#{i+1}** → {item['resultado']} | Confiança: {item['confianca']}")
