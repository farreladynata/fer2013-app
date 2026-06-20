import streamlit as st
import numpy as np
from PIL import Image
import tf_keras as tf

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Ekspresi Wajah",
    page_icon="😊",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .title-block {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .title-block h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.25rem;
    }
    .title-block p {
        color: #8b8fa8;
        font-size: 0.95rem;
    }

    .result-card {
        background: #1c1f2e;
        border: 1px solid #2e3150;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-card .emoji { font-size: 3rem; }
    .result-card .label {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.5rem 0 0.25rem 0;
    }
    .result-card .confidence {
        color: #8b8fa8;
        font-size: 0.9rem;
    }

    .bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        color: #8b8fa8;
        margin-bottom: 2px;
    }

    .upload-hint {
        color: #8b8fa8;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 0.5rem;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_EMOJI  = ['😠', '🤢', '😨', '😄', '😐', '😢', '😲']
IMG_SIZE = 48

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_fer2013_cnn.h5')

model = load_model()

# ── Helper ────────────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert('L').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype='float32') / 255.0
    return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>😊 Klasifikasi Ekspresi Wajah</h1>
    <p>Upload foto wajah — model CNN akan mendeteksi ekspresi secara otomatis</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Pilih gambar wajah",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown('<p class="upload-hint">Format yang didukung: JPG, JPEG, PNG</p>', unsafe_allow_html=True)

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Gambar yang diupload", use_container_width=True)

    with col2:
        with st.spinner("Menganalisis ekspresi..."):
            x = preprocess(image)
            preds = model.predict(x)[0]

        top_idx   = int(np.argmax(preds))
        top_label = EMOTION_LABELS[top_idx]
        top_emoji = EMOTION_EMOJI[top_idx]
        top_conf  = float(preds[top_idx]) * 100

        st.markdown(f"""
        <div class="result-card">
            <div class="emoji">{top_emoji}</div>
            <div class="label">{top_label}</div>
            <div class="confidence">Confidence: {top_conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Probabilitas per Kelas")
    sorted_idx = np.argsort(preds)[::-1]
    for i in sorted_idx:
        label = EMOTION_LABELS[i]
        prob  = float(preds[i])
        emoji = EMOTION_EMOJI[i]
        col_l, col_b = st.columns([1, 4])
        with col_l:
            st.markdown(f"**{emoji} {label}**")
        with col_b:
            st.progress(prob, text=f"{prob*100:.1f}%")

else:
    st.info("Upload gambar wajah di atas untuk memulai klasifikasi.")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#8b8fa8; font-size:0.8rem;'>"
    "Model: CNN · Dataset: FER2013 · 7 kelas ekspresi</p>",
    unsafe_allow_html=True
)
