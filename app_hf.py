import streamlit as st
import torch
from transformers import pipeline
import os
import base64
import gc

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Cloudera GenAI Demo", layout="wide")

# --- VARIABLES DE ESTADO ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- MODELOS ACTUALIZADOS CON MÁS DETALLE ---
WHISPER_MODELS = {
    "🚀 Rápido | Whisper Small (244M) - Ideal demos rápidas": "openai/whisper-small",
    "⚖️ Equilibrado | Whisper Medium (769M) - Mejor precisión": "openai/whisper-medium",
    "🧠 Potente | Whisper Large v3 (1.5B) - Calidad humana (Req. GPU >10GB)": "openai/whisper-large-v3"
}

NLLB_MODELS = {
    "⚡ Veloz | NLLB Distilled (600M) - Traducción ágil": "facebook/nllb-200-distilled-600M",
    "🎯 Preciso | NLLB (1.3B) - Gramática compleja y matices": "facebook/nllb-200-1.3B"
}

LANG_CONFIG = {
    "Español": {"whisper": "spanish", "nllb": "spa_Latn"},
    "Inglés":  {"whisper": "english", "nllb": "eng_Latn"},
    "Árabe":   {"whisper": "arabic",  "nllb": "ara_Arab"},
    "Francés": {"whisper": "french",  "nllb": "fra_Latn"},
    "Alemán":  {"whisper": "german",  "nllb": "deu_Latn"},
    "Chino":   {"whisper": "chinese", "nllb": "zho_Hans"},
    "Ruso":    {"whisper": "russian", "nllb": "rus_Cyrl"}, # Añadido extra
    "Japonés": {"whisper": "japanese", "nllb": "jpn_Jpan"}, # Añadido extra
}

# --- CSS DEFINITIVO (DISEÑO BLANCO / TEXTO VERDE) ---
def get_base64_of_bin_file(bin_file):
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

    /* 7. PESTAÑAS (GRIS ACTIVO) */
    button[data-baseweb="tab"] {{ 
        color: {corp_green} !important; 
        background-color: transparent !important;
        border: 1px solid transparent;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        background-color: {active_grey} !important;
        color: white !important;
        border-radius: 5px;
        border: 1px solid {active_grey} !important;
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
        color: rgb(52, 123, 94) !important; /* Texto verde en footer */
        border-top: 1px solid rgb(52, 123, 94); /* Línea verde de separación */
        text-align: center; padding: 10px; z-index: 1000;
        display: flex; align-items: center; justify-content: center;
    }}
    .footer span {{ color: rgb(52, 123, 94) !important; font-weight: bold; }}
    [data-testid="stAppViewContainer"] > section:first-child {{ padding-bottom: 80px; }}
    </style>
    <div class="footer"><span>@2026</span>{img_tag}</div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


def render_header(logo_path="logo.png", robot_path="robot.png"):
    # 1. Procesar Logo (Doble que antes -> 900px)
    logo_html = ""
    if os.path.exists(logo_path):
        logo_b64 = get_base64_of_bin_file(logo_path)
        # width: 900px (antes 450px). Añadimos max-width 100% para que no se salga en pantallas pequeñas
        logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="width: 900px; max-width: 100%; margin-bottom: 20px;">'
    
    # 2. Procesar Robot (Triple que antes -> 900px)
    robot_html = ""
    if os.path.exists(robot_path):
        robot_b64 = get_base64_of_bin_file(robot_path)
        # width: 900px (antes 200px)
        robot_html = f'<img src="data:image/png;base64,{robot_b64}" style="width: 900px; max-width: 100%; margin-bottom: 25px;">'

    # 3. Renderizar Contenedor Centrado
    header_html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; margin-bottom: 40px;">
        {logo_html}
        {robot_html}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    


# --- CARGA (CACHE) ---
@st.cache_resource(show_spinner="Iniciando motores de IA...")
def load_whisper(model_id):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return pipeline("automatic-speech-recognition", model=model_id, device=device)

@st.cache_resource(show_spinner="Preparando traductores...")
def load_translator(model_id):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return pipeline("translation", model=model_id, device=device)


# --- INICIO ---
# 1. Cargar Estilos CSS (sin pintar logo antiguo)
set_custom_style() 

# 2. Pintar la Cabecera Nueva (Logo Grande + Robot + Titulo)
# Asegúrate de tener los archivos 'logo.png' y 'robot.png' en la carpeta
render_header("logo.png", "robot.png") 

# 3. Footer
add_footer("cloudera-logo.png")

# Sidebar (Se mantiene igual...)
with st.sidebar:
    st.header("⚙️ Configuración")
    w_name = st.selectbox("Modelo Audio", list(WHISPER_MODELS.keys()))
    n_name = st.selectbox("Modelo Traducción", list(NLLB_MODELS.keys()))
    
    if st.button("🧹 Limpiar Memoria"):
        torch.cuda.empty_cache()
        gc.collect()
        st.toast("Memoria liberada")

# Cargar modelos
try:
    whisper_pipe = load_whisper(WHISPER_MODELS[w_name])
    translator_pipe = load_translator(NLLB_MODELS[n_name])
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- UI PRINCIPAL ---
st.title("Demo de Cloudera: GenAI on-premise")

# ==========================================
# SECCIÓN 1: TRADUCCIÓN BATCH
# ==========================================
st.markdown("---")
st.header("📂 1. Traducción Batch")
st.markdown("Sube ficheros de audio o realiza grabaciones largas.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Entrada")
    input_lang_batch = st.selectbox("Idioma Origen", list(LANG_CONFIG.keys()), key="batch_in")
    
    tab_mic, tab_file = st.tabs(["Grabar", "Subir Fichero"])
    audio_bytes_batch = None
    with tab_mic:
        m = st.audio_input("Grabar mensaje largo...", key="mic_batch")
        if m: audio_bytes_batch = m.read()
    with tab_file:
        f = st.file_uploader("Subir MP3/WAV", type=["mp3","wav","m4a"], key="file_batch")
        if f: audio_bytes_batch = f.read()

with col2:
    st.subheader("Resultado")
    target_lang_batch = st.selectbox("Idioma Destino", list(LANG_CONFIG.keys()), index=1, key="batch_out")
    
    if audio_bytes_batch:
        with st.spinner("Procesando Batch..."):
            res = whisper_pipe(audio_bytes_batch, generate_kwargs={"language": LANG_CONFIG[input_lang_batch]["whisper"]})
            text_batch = res["text"]
            trans = translator_pipe(text_batch, src_lang=LANG_CONFIG[input_lang_batch]["nllb"], tgt_lang=LANG_CONFIG[target_lang_batch]["nllb"])
            final_batch = trans[0]['translation_text']
        
        st.info(f"Transcripción: {text_batch}")
        st.success(f"Traducción: {final_batch}")


# ==========================================
# SECCIÓN 2: TRADUCCIÓN SIMULTÁNEA / CHAT
# ==========================================
st.markdown("---")
st.header("⚡ 2. Traducción Simultánea")
st.markdown("Habla frases cortas. El sistema irá traduciendo en tiempo real.")

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    chat_in_lang = st.selectbox("Hablo en:", list(LANG_CONFIG.keys()), key="chat_in")
with c2:
    chat_out_lang = st.selectbox("Traducir a:", list(LANG_CONFIG.keys()), index=1, key="chat_out")
with c3:
    if st.button("🗑️ Borrar Historial"):
        st.session_state.chat_history = []
        st.rerun()

audio_chat = st.audio_input("💬 Habla ahora (frase corta)...", key="mic_chat")

if audio_chat:
    b = audio_chat.read()
    with st.spinner("Analizando..."):
        w_code = LANG_CONFIG[chat_in_lang]["whisper"]
        t_text = whisper_pipe(b, generate_kwargs={"language": w_code})["text"]
        src = LANG_CONFIG[chat_in_lang]["nllb"]
        tgt = LANG_CONFIG[chat_out_lang]["nllb"]
        tr_text = translator_pipe(t_text, src_lang=src, tgt_lang=tgt)[0]['translation_text']
        
        st.session_state.chat_history.append({
            "original": t_text,
            "translated": tr_text,
            "lang": chat_out_lang
        })

st.markdown("### 📝 Historial")
chat_container = st.container(height=400)

with chat_container:
    if not st.session_state.chat_history:
        st.caption("El historial está vacío.")
    
    for i, msg in enumerate(reversed(st.session_state.chat_history)):
        with st.chat_message("user", avatar="👮‍♂️"):
            # Para el chat, usamos un fondo gris claro para que contraste con el blanco de fondo
            # y el texto en verde corporativo.
            
            st.markdown(f"**Origen:** {msg['original']}")
            st.markdown(f"**Traducción ({msg['lang']}):**")
            
            # Estilo del bocadillo
            bubble_style = """
                background-color: #f0f2f6; 
                color: rgb(52, 123, 94); 
                padding: 10px; 
                border-radius: 10px; 
                border: 1px solid #e0e0e0;
            """
            
            if msg['lang'] == "Árabe":
                st.markdown(f'<div style="{bubble_style} text-align: right;">{msg["translated"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="{bubble_style}">{msg["translated"]}</div>', unsafe_allow_html=True)