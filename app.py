import streamlit as st
import torch
from transformers import pipeline
import os
import base64
import gc
import scipy.io.wavfile
import numpy as np
from io import BytesIO

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Cloudera GenAI Demo", layout="wide")

# --- VARIABLES DE ESTADO ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- MODELOS ACTUALIZADOS (SOLO LOS LIGEROS) ---
WHISPER_MODELS = {
    "🚀 Rápido | Whisper Small (244M)": "openai/whisper-small"
}

NLLB_MODELS = {
    "⚡ Veloz | NLLB Distilled (600M)": "facebook/nllb-200-distilled-600M"
}

# Añadimos los modelos MMS-TTS de Meta (Offline y super ligeros ~140MB)
LANG_CONFIG = {
    "Español": {"whisper": "spanish", "nllb": "spa_Latn", "tts": "facebook/mms-tts-spa"},
    "Inglés":  {"whisper": "english", "nllb": "eng_Latn", "tts": "facebook/mms-tts-eng"},
    "Catalán": {"whisper": "catalan", "nllb": "cat_Latn", "tts": "facebook/mms-tts-cat"},
    "Euskera": {"whisper": "basque",  "nllb": "eus_Latn", "tts": "facebook/mms-tts-eus"},
    "Gallego": {"whisper": "galician", "nllb": "glg_Latn", "tts": "facebook/mms-tts-glg"},
    "Valenciano": {"whisper": "catalan", "nllb": "cat_Latn", "tts": "facebook/mms-tts-cat"},
    "Árabe":   {"whisper": "arabic",  "nllb": "ara_Arab", "tts": "facebook/mms-tts-ara"},
    "Francés": {"whisper": "french",  "nllb": "fra_Latn", "tts": "facebook/mms-tts-fra"},
    "Alemán":  {"whisper": "german",  "nllb": "deu_Latn", "tts": "facebook/mms-tts-deu"},
    "Chino":   {"whisper": "chinese", "nllb": "zho_Hans", "tts": "facebook/mms-tts-cmn"},
    "Ruso":    {"whisper": "russian", "nllb": "rus_Cyrl", "tts": "facebook/mms-tts-rus"}, 
    "Japonés": {"whisper": "japanese", "nllb": "jpn_Jpan", "tts": "facebook/mms-tts-jpn"}, 
}

# --- CSS DEFINITIVO (DISEÑO BLANCO / TEXTO VERDE) ---
def get_base64_of_bin_file(bin_file):
    if not os.path.exists(bin_file):
        return ""
    with open(bin_file, 'rb') as f: data = f.read()
    return base64.b64encode(data).decode()

def set_custom_style():
    corp_green = "rgb(52, 123, 94)"
    active_grey = "#dddddd"
    
    css = f"""
    <style>
    /* 1. FONDO GENERAL */
    [data-testid="stAppViewContainer"] {{ background-color: white; }}
    [data-testid="stSidebar"] {{ background-color: #f8f9fa; border-right: 1px solid #ddd; }}
    
    /* 2. HEADER */
    header[data-testid="stHeader"] {{ background-color: white !important; }}
    header[data-testid="stHeader"] svg {{ fill: {corp_green} !important; }}
    
    /* 3. TIPOGRAFÍA */
    h1, h2, h3, h4, p, label, .stMarkdown, span, div, li {{ 
        color: {corp_green} !important; 
    }}
    
    /* 4. SEPARADORES */
    hr {{ border-color: {corp_green} !important; }}

    /* 5. WIDGETS */
    div[data-baseweb="select"] > div {{
        background-color: white !important;
        color: {corp_green} !important;
        border-color: {corp_green} !important;
        font-weight: bold;
    }}
    [data-testid="stFileUploader"] {{ color: {corp_green} !important; }}
    
    /* 6. BOTONES */
    div.stButton > button {{
        color: {corp_green} !important;
        background-color: white !important;
        border: 2px solid {corp_green} !important;
        font-weight: bold !important;
        border-radius: 5px;
    }}
    div.stButton > button:hover {{
        background-color: {corp_green} !important;
        color: white !important;
    }}

    /* 8. ALERTAS */
    div[data-testid="stStatusWidget"] div, 
    div[data-testid="stAlert"] div {{
        color: {corp_green} !important; 
    }}
    div[data-testid="stAlert"] {{
        background-color: #f0fdf4 !important;
        border: 1px solid {corp_green} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def add_footer(footer_logo_path="cloudera-logo.png"):
    img_tag = ""
    if os.path.exists(footer_logo_path):
        footer_img_b64 = get_base64_of_bin_file(footer_logo_path)
        img_tag = f'<img src="data:image/png;base64,{footer_img_b64}" style="height: 30px; margin-left: 15px;">'
    
    footer_html = f"""
    <style>
    .footer {{
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: white; 
        color: rgb(52, 123, 94) !important;
        border-top: 1px solid rgb(52, 123, 94);
        text-align: center; padding: 10px; z-index: 1000;
        display: flex; align-items: center; justify-content: center;
    }}
    .footer span {{ color: rgb(52, 123, 94) !important; font-weight: bold; }}
    [data-testid="stAppViewContainer"] > section:first-child {{ padding-bottom: 80px; }}
    </style>
    <div class="footer"><span>@2026 Cloudera AI</span>{img_tag}</div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

def render_header(image_path="traductor.png"):
    image_html = ""
    if os.path.exists(image_path):
        image_b64 = get_base64_of_bin_file(image_path)
        image_html = f'<img src="data:image/png;base64,{image_b64}" style="max-width: 100%; height: auto; margin-bottom: 20px;">'
    
    header_html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; margin-bottom: 40px;">
        {image_html}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

# --- CARGA (CACHE) ---
@st.cache_resource(show_spinner="Iniciando motor de Audio (Whisper)...")
def load_whisper(model_id):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("automatic-speech-recognition", model=model_id, device=device)

@st.cache_resource(show_spinner="Preparando Traductor (NLLB)...")
def load_translator(model_id):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("translation_en_to_es", model=model_id, device=device, src_lang="eng_Latn", tgt_lang="spa_Latn")

@st.cache_resource(show_spinner="Cargando Sintetizador de Voz (MMS Offline)...")
def load_tts(model_id):
    # El TTS lo cargamos bajo demanda según el idioma que elijan.
    # Al ser muy ligero, funciona rapidísimo.
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-to-speech", model=model_id, device=device)

# --- INICIO ---
set_custom_style() 
render_header("traductor.png") 
add_footer("cloudera-logo.png")

with st.sidebar:
    st.header("⚙️ Configuración")
    w_name = st.selectbox("Modelo Audio", list(WHISPER_MODELS.keys()))
    n_name = st.selectbox("Modelo Traducción", list(NLLB_MODELS.keys()))
    
    if st.button("🧹 Limpiar Memoria"):
        torch.cuda.empty_cache()
        gc.collect()
        st.toast("Memoria liberada")

# Cargar modelos principales
try:
    whisper_pipe = load_whisper(WHISPER_MODELS[w_name])
    translator_pipe = load_translator(NLLB_MODELS[n_name])
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- UI PRINCIPAL ---
st.title("Demo de Cloudera: GenAI on-premise")

# ==========================================
# SECCIÓN ÚNICA: TRADUCCIÓN Y SÍNTESIS BATCH
# ==========================================
st.markdown("---")
st.header("🎙️ Traductor y Lector Universal (100% Offline)")
st.markdown("Graba un mensaje en tu idioma. El sistema lo traducirá y lo leerá usando modelos locales.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Entrada")
    input_lang_batch = st.selectbox("Idioma Origen", list(LANG_CONFIG.keys()), key="batch_in")
    
    tab_mic, tab_file = st.tabs(["Grabar", "Subir Fichero"])
    audio_bytes_batch = None
    with tab_mic:
        m = st.audio_input("Grabar mensaje...", key="mic_batch")
        if m: audio_bytes_batch = m.read()
    with tab_file:
        f = st.file_uploader("Subir MP3/WAV", type=["mp3","wav","m4a"], key="file_batch")
        if f: audio_bytes_batch = f.read()

with col2:
    st.subheader("Resultado")
    target_lang_batch = st.selectbox("Idioma Destino", list(LANG_CONFIG.keys()), index=1, key="batch_out")
    
    if audio_bytes_batch:
        with st.spinner("Transcribiendo y Traduciendo..."):
            # 1. Transcripción Whisper
            res = whisper_pipe(audio_bytes_batch, generate_kwargs={"language": LANG_CONFIG[input_lang_batch]["whisper"]})
            text_batch = res["text"]
            
            # 2. Traducción NLLB
            trans = translator_pipe(text_batch, src_lang=LANG_CONFIG[input_lang_batch]["nllb"], tgt_lang=LANG_CONFIG[target_lang_batch]["nllb"])
            final_batch = trans[0]['translation_text']
        
        st.info(f"**Transcripción original:** {text_batch}")
        st.success(f"**Traducción:** {final_batch}")
        
        # 3. Text to Speech OFFLINE (Generación de voz con Hugging Face MMS)
        with st.spinner("Generando audio de voz (Offline)..."):
            try:
                # Cargar el pipeline de voz específico para el idioma seleccionado
                tts_model_id = LANG_CONFIG[target_lang_batch]["tts"]
                tts_pipe = load_tts(tts_model_id)
                
                # Generar el audio
                audio_out = tts_pipe(final_batch)
                audio_array = audio_out["audio"][0] # Array numpy con las ondas
                sample_rate = audio_out["sampling_rate"]
                
                # Guardar el audio a formato WAV en memoria RAM
                fp = BytesIO()
                scipy.io.wavfile.write(fp, sample_rate, audio_array)
                
                st.markdown("🔊 **Escucha la traducción:**")
                st.audio(fp.getvalue(), format='audio/wav')
                
            except Exception as e:
                st.error(f"Detalle técnico TTS: {str(e)}")