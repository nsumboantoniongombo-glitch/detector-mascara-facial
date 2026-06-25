# 😷 Detector de Máscara Facial — Deep Learning

Sistema de detecção automática de máscara facial usando Deep Learning com Transfer Learning (MobileNetV2).

## 🚀 Aplicação Online

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://detector-mascara.streamlit.app)

## 🤖 Tecnologia

- **Arquitectura:** MobileNetV2 (Transfer Learning)
- **Framework:** TensorFlow / Keras
- **Interface:** Streamlit
- **Tipo:** Deep Learning — Classificação Binária

## 📊 Funcionamento

O modelo analisa imagens e classifica em duas categorias:
- ✅ **Com Máscara** — pessoa usando máscara facial
- ❌ **Sem Máscara** — pessoa sem máscara facial

## 🛠️ Como Executar Localmente

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## 📁 Estrutura

```
detector-mascara/
├── dashboard.py          → Interface Streamlit
├── treino.py             → Treino do modelo CNN
├── detector.py           → Detecção via webcam
├── preprocessamento.py   → Pré-processamento de imagens
├── avaliacao.py          → Avaliação do modelo
├── modelo/
│   └── modelo_mascara.h5 → Modelo treinado
├── dataset/
│   ├── com_mascara/      → Imagens de treino
│   └── sem_mascara/      → Imagens de treino
└── assets/
    └── grafico_treino.png
```

## 👥 Universidade Kimpa Vita
Escola Superior Politécnico do Uíge — Engenharia Informática — 2026
