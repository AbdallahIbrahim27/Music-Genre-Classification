import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
import io
import time
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beat Analyzer — Music Genre Detector",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Root reset ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Dark background ── */
.stApp {
    background: #0a0a0f;
    color: #e8e6f0;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 760px; }

/* ── Hero title ── */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: -2px;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1rem;
    color: #6b6880;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #13121a !important;
    border: 1.5px dashed #2e2c3e !important;
    border-radius: 16px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6c63ff !important;
}
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #13121a 0%, #1a1826 100%);
    border: 1px solid #2e2c3e;
    border-radius: 20px;
    padding: 2rem 2.2rem;
    margin-top: 1.5rem;
}
.genre-label {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}
.confidence-text {
    font-size: 0.85rem;
    color: #6b6880;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1.5rem;
}

/* ── Progress bars ── */
.bar-row { margin-bottom: 0.55rem; }
.bar-label {
    font-size: 0.78rem;
    color: #9d9ab0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    display: flex;
    justify-content: space-between;
    margin-bottom: 3px;
}
.bar-track {
    background: #1e1c2a;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
}

/* ── Info pill ── */
.info-pill {
    display: inline-block;
    background: #1e1c2a;
    border: 1px solid #2e2c3e;
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-size: 0.78rem;
    color: #9d9ab0;
    margin-right: 0.5rem;
    margin-top: 0.4rem;
    font-family: 'Space Mono', monospace;
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #4a4860;
    margin-bottom: 0.8rem;
    margin-top: 1.8rem;
}

/* ── Waveform / spectrogram container ── */
.viz-container {
    background: #0e0d14;
    border: 1px solid #1e1c2a;
    border-radius: 12px;
    overflow: hidden;
    margin-top: 0.5rem;
}

/* ── Model selector ── */
[data-testid="stSelectbox"] > div {
    background: #13121a !important;
    border-color: #2e2c3e !important;
    color: #e8e6f0 !important;
    border-radius: 10px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #6c63ff;
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.6rem 1.5rem;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #7d75ff;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(108, 99, 255, 0.35);
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #6c63ff !important; }

/* ── Divider ── */
hr { border-color: #1e1c2a; margin: 2rem 0; }

/* ── Error / warning ── */
[data-testid="stAlert"] {
    background: #1a1018 !important;
    border-color: #3d1f2a !important;
    border-radius: 12px !important;
    color: #e8a0b0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Genre metadata ─────────────────────────────────────────────────────────────
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

GENRE_META = {
    'blues':     {'emoji': '🎸', 'color': '#3b82f6', 'desc': 'Soulful, emotional, root of rock'},
    'classical': {'emoji': '🎻', 'color': '#a78bfa', 'desc': 'Orchestral, structured, timeless'},
    'country':   {'emoji': '🤠', 'color': '#f59e0b', 'desc': 'Storytelling, twang, heartland'},
    'disco':     {'emoji': '🪩', 'color': '#ec4899', 'desc': 'Groovy, danceable, 70s funk'},
    'hiphop':    {'emoji': '🎤', 'color': '#f97316', 'desc': 'Rhythmic, lyrical, urban culture'},
    'jazz':      {'emoji': '🎷', 'color': '#10b981', 'desc': 'Improvised, complex, sophisticated'},
    'metal':     {'emoji': '🤘', 'color': '#ef4444', 'desc': 'Heavy, distorted, intense energy'},
    'pop':       {'emoji': '🎀', 'color': '#f472b6', 'desc': 'Catchy, mainstream, polished'},
    'reggae':    {'emoji': '🌿', 'color': '#22c55e', 'desc': 'Laid-back, Jamaican, offbeat rhythm'},
    'rock':      {'emoji': '⚡', 'color': '#e879f9', 'desc': 'Powerful, guitar-driven, energetic'},
}

# ── Feature extraction (same as tabular_approach.py) ─────────────────────────
def extract_features(y, sr):
    features = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
        features[f'mfcc_{i+1}_std']  = float(np.std(mfcc[i]))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = float(np.mean(chroma))
    features['chroma_std']  = float(np.std(chroma))
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = float(np.mean(sc))
    features['spectral_centroid_std']  = float(np.std(sc))
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff_mean'] = float(np.mean(ro))
    features['spectral_rolloff_std']  = float(np.std(ro))
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std']  = float(np.std(zcr))
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std']  = float(np.std(rms))
    return features

# ── Spectrogram image ──────────────────────────────────────────────────────────
def make_melspec_image(y, sr, figsize=(8, 3)):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0e0d14')
    ax.set_facecolor('#0e0d14')
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
                              ax=ax, cmap='inferno')
    ax.tick_params(colors='#4a4860', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e1c2a')
    ax.set_xlabel('Time (s)', color='#4a4860', fontsize=8)
    ax.set_ylabel('Hz', color='#4a4860', fontsize=8)
    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, facecolor='#0e0d14')
    plt.close(fig)
    buf.seek(0)
    return buf

def make_waveform_image(y, sr, figsize=(8, 2)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0e0d14')
    ax.set_facecolor('#0e0d14')
    times = np.linspace(0, len(y) / sr, len(y))
    ax.fill_between(times, y, alpha=0.6, color='#6c63ff')
    ax.plot(times, y, color='#9d97ff', linewidth=0.4, alpha=0.8)
    ax.set_xlim(0, times[-1])
    ax.tick_params(colors='#4a4860', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e1c2a')
    ax.set_xlabel('Time (s)', color='#4a4860', fontsize=8)
    ax.set_ylabel('Amplitude', color='#4a4860', fontsize=8)
    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, facecolor='#0e0d14')
    plt.close(fig)
    buf.seek(0)
    return buf

# ── Demo prediction (no saved model needed) ───────────────────────────────────
def predict_from_features(features_dict, genre_list):
    """
    Heuristic demo predictor based on extracted audio features.
    Replace with joblib.load('tabular_best_model.pkl') when model is trained.
    """
    zcr   = features_dict.get('zcr_mean', 0)
    rms   = features_dict.get('rms_mean', 0)
    cent  = features_dict.get('spectral_centroid_mean', 0)
    chrom = features_dict.get('chroma_mean', 0)
    mfcc1 = features_dict.get('mfcc_1_mean', 0)
    mfcc2 = features_dict.get('mfcc_2_mean', 0)

    scores = {g: 0.0 for g in genre_list}

    # Energy-based
    if rms > 0.12:
        scores['metal'] += 3; scores['rock'] += 2; scores['disco'] += 1
    elif rms > 0.07:
        scores['hiphop'] += 2; scores['pop'] += 2; scores['reggae'] += 1
    else:
        scores['classical'] += 3; scores['jazz'] += 2; scores['blues'] += 1

    # ZCR (zero-crossings: high = metal/rock, low = bass-heavy)
    if zcr > 0.12:
        scores['metal'] += 2; scores['rock'] += 1
    elif zcr > 0.07:
        scores['pop'] += 1; scores['country'] += 1
    else:
        scores['hiphop'] += 2; scores['reggae'] += 1; scores['blues'] += 1

    # Spectral centroid (brightness)
    if cent > 3000:
        scores['metal'] += 1; scores['classical'] += 1
    elif cent > 1800:
        scores['pop'] += 1; scores['rock'] += 1; scores['disco'] += 1
    else:
        scores['hiphop'] += 1; scores['reggae'] += 2; scores['jazz'] += 1

    # Chroma (harmonic content)
    if chrom > 0.5:
        scores['jazz'] += 2; scores['classical'] += 1; scores['blues'] += 1
    else:
        scores['metal'] += 1; scores['hiphop'] += 1

    # MFCC1 (overall energy shape)
    if mfcc1 < -200:
        scores['classical'] += 2
    elif mfcc1 < -100:
        scores['jazz'] += 1; scores['blues'] += 1
    else:
        scores['hiphop'] += 1; scores['pop'] += 1

    # Softmax-like normalization
    vals = np.array([scores[g] for g in genre_list], dtype=float)
    vals = np.clip(vals + np.random.uniform(0, 0.5, len(vals)), 0.1, None)
    probs = np.exp(vals) / np.exp(vals).sum()
    return probs

# ── Load saved model if it exists ─────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        bundle = joblib.load(path)
        return bundle['model'], bundle['label_encoder'], bundle['feature_cols']
    return None, None, None

# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Beat Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Music Genre Detection · AI-powered</div>', unsafe_allow_html=True)

# ── Model status ──────────────────────────────────────────────────────────────
model, le, feature_cols = load_model('tabular_best_model.pkl')
if model:
    st.markdown(
        '<span class="info-pill">✓ Trained model loaded</span>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<span class="info-pill">⚡ Demo mode — heuristic predictor</span>'
        '<span class="info-pill">Train model for higher accuracy</span>',
        unsafe_allow_html=True
    )

st.markdown('<div style="margin-top:1.8rem"></div>', unsafe_allow_html=True)

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop a .wav or .mp3 audio file here",
    type=['wav', 'mp3'],
    help="Upload any audio clip — ideally 5–30 seconds long",
    label_visibility='collapsed',
)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center;margin-top:3rem;color:#2e2c3e;">
        <div style="font-size:3rem;margin-bottom:0.5rem">🎵</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.75rem;letter-spacing:0.1em;color:#2e2c3e">
            UPLOAD AN AUDIO FILE TO BEGIN
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Audio playback ────────────────────────────────────────────────────────────
st.audio(uploaded)

# ── Load & process ────────────────────────────────────────────────────────────
with st.spinner("Analysing audio..."):
    try:
        audio_bytes = uploaded.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=30, mono=True)
    except Exception as e:
        st.error(f"Could not load audio: {e}")
        st.stop()

    duration  = len(y) / sr
    tempo, _  = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    features = extract_features(y, sr)

    # Predict
    if model and feature_cols:
        X = np.array([[features.get(c, 0.0) for c in feature_cols]])
        pred_idx = model.predict(X)[0]
        pred_genre = le.inverse_transform([pred_idx])[0]
        try:
            probs = model.predict_proba(X)[0]
            genre_probs = {le.classes_[i]: probs[i] for i in range(len(le.classes_))}
        except:
            probs = predict_from_features(features, GENRES)
            genre_probs = {GENRES[i]: probs[i] for i in range(len(GENRES))}
    else:
        probs = predict_from_features(features, GENRES)
        genre_probs = {GENRES[i]: probs[i] for i in range(len(GENRES))}
        pred_genre = max(genre_probs, key=genre_probs.get)

    confidence   = genre_probs[pred_genre]
    meta         = GENRE_META[pred_genre]
    color        = meta['color']
    sorted_genres = sorted(genre_probs.items(), key=lambda x: -x[1])

# ── Result card ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="result-card">
  <div class="genre-label" style="color:{color}">{meta['emoji']} {pred_genre.upper()}</div>
  <div class="confidence-text">{meta['desc']}</div>
</div>
""", unsafe_allow_html=True)



# ── Audio stats ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Audio stats</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Duration",  f"{duration:.1f}s")
c2.metric("BPM",       f"{tempo_val:.0f}")
c3.metric("Sample rate", f"{sr//1000}kHz")
c4.metric("RMS energy", f"{features['rms_mean']:.3f}")

# ── Waveform ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Waveform</div>', unsafe_allow_html=True)
with st.container():
    wave_buf = make_waveform_image(y, sr)
    st.image(wave_buf, use_container_width=True)

# ── Mel spectrogram ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Mel spectrogram</div>', unsafe_allow_html=True)
with st.container():
    spec_buf = make_melspec_image(y, sr)
    st.image(spec_buf, use_container_width=True)

# ── MFCC heatmap ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">MFCC coefficients</div>', unsafe_allow_html=True)
mfcc_full = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
fig, ax = plt.subplots(figsize=(8, 2.2), facecolor='#0e0d14')
ax.set_facecolor('#0e0d14')
librosa.display.specshow(mfcc_full, sr=sr, x_axis='time', ax=ax, cmap='coolwarm')
ax.set_ylabel('MFCC', color='#4a4860', fontsize=8)
ax.set_xlabel('Time (s)', color='#4a4860', fontsize=8)
ax.tick_params(colors='#4a4860', labelsize=7)
for spine in ax.spines.values():
    spine.set_edgecolor('#1e1c2a')
plt.tight_layout(pad=0.4)
buf3 = io.BytesIO()
plt.savefig(buf3, format='png', dpi=130, facecolor='#0e0d14')
plt.close(fig)
buf3.seek(0)
st.image(buf3, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center;font-size:0.72rem;color:#2e2c3e;font-family:'Space Mono',monospace;letter-spacing:0.08em">
    BEATANALYZER · GTZAN · 10 GENRES · Abdullah Ibrahim
</div>
""", unsafe_allow_html=True)