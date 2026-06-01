"""
BIL216 - İşaretler ve Sistemler | Final Proje
2025-2026 Bahar Dönemi | Grup 10
Duygu Sınıflandırma - Faz 3
"""

import os
import glob
import warnings
import tempfile
import queue
import threading

import numpy as np
import pandas as pd
import librosa
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st

from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ── streamlit-webrtc (mikrofon için) ──────────────────────────────────────────
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
    import av
    WEBRTC_OK = True
except ImportError:
    WEBRTC_OK = False

warnings.filterwarnings("ignore")
np.random.seed(42)

st.set_page_config(
    page_title="Duygu Sınıflandırma | Grup 10 | Faz 3",
    page_icon="🌸",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500;600;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Palatino Linotype', 'Palatino', 'Book Antiqua', 'EB Garamond', Georgia, serif !important;
}

.stApp {
    background: linear-gradient(135deg,
        #fce4ec 0%,
        #f8e6f0 15%,
        #e8d5f0 30%,
        #dce8f8 50%,
        #e0eeff 65%,
        #ecd6f5 80%,
        #fde8f0 100%
    ) !important;
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: rgba(255, 240, 248, 0.88) !important;
    backdrop-filter: blur(16px);
    border-right: 1.5px solid #f0b8d4 !important;
}
[data-testid="stSidebar"] * { color: #5a3050 !important; }

.stApp { color: #4a2d5a; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] b,
[data-testid="stMarkdownContainer"] i,
.stMarkdown p, .stMarkdown li,
[data-testid="stText"],
label { color: #4a2d5a !important; }

[data-testid="stHeading"] { color: #4a2d5a !important; }

[data-testid="stMetric"] {
    background: rgba(255, 248, 252, 0.92) !important;
    border: 1.5px solid #e8b4d4 !important;
    border-radius: 20px !important;
    padding: 16px 20px !important;
    box-shadow: 0 6px 20px rgba(200, 100, 160, 0.12) !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: #b5478a !important;
    font-family: 'Palatino Linotype', Georgia, serif !important;
}
[data-testid="stMetricLabel"] {
    color: #c888b0 !important;
    font-size: .82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 248, 252, 0.80) !important;
    border-radius: 16px !important;
    padding: 5px !important;
    gap: 4px !important;
    border: 1.5px solid #e8b4d4 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px !important;
    color: #c878b0 !important;
    font-weight: 600 !important;
    font-family: 'Palatino Linotype', Georgia, serif !important;
    font-size: 0.92rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #e8a0c8, #b8a0e0, #a0c0e8) !important;
    color: white !important;
    box-shadow: 0 3px 14px rgba(180, 120, 200, 0.40) !important;
}

.stButton > button {
    background: linear-gradient(90deg, #e8a0c8, #c0a0e0, #a0b8e8) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    font-family: 'Palatino Linotype', Georgia, serif !important;
    padding: 10px 30px !important;
    box-shadow: 0 4px 18px rgba(180, 120, 200, 0.35) !important;
    transition: all .25s ease !important;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(180, 120, 200, 0.50) !important;
}

.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: rgba(255, 248, 252, 0.95) !important;
    border: 1.5px solid #e8b4d4 !important;
    border-radius: 12px !important;
    color: #4a2d5a !important;
    font-weight: 500 !important;
    font-family: 'Palatino Linotype', Georgia, serif !important;
}

.stSlider [data-baseweb="slider"] { color: #c878b0 !important; }
hr { border-color: #e8b4d4 !important; }

.stTable thead tr th {
    background: linear-gradient(90deg, #f5d0e8, #e8d0f5) !important;
    color: #5a2060 !important;
    font-weight: 700 !important;
    font-family: 'Palatino Linotype', Georgia, serif !important;
}
.stDataFrame { border-radius: 12px !important; overflow: hidden; }
.stProgress > div > div { background: linear-gradient(90deg, #e8a0c8, #b8a0e0) !important; }
.stRadio label { color: #5a3060 !important; font-weight: 600 !important; }
[data-testid="stAlert"] {
    border-radius: 14px !important;
    font-family: 'Palatino Linotype', Georgia, serif !important;
}

.hero-wrap {
    background: rgba(255, 245, 252, 0.75);
    border: 1.5px solid #f0c0dc;
    border-radius: 22px;
    padding: 24px 32px 18px;
    margin-bottom: 8px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 30px rgba(200, 100, 160, 0.10);
}
.hero-title {
    background: linear-gradient(90deg, #c05090, #9060c0, #5090d0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0;
    font-family: 'Palatino Linotype', Georgia, serif !important;
    letter-spacing: -0.01em;
}
.hero-sub {
    color: #c878b0 !important;
    font-size: .9rem;
    font-weight: 500;
    margin-top: 4px;
    letter-spacing: 0.04em;
    font-family: 'Palatino Linotype', Georgia, serif !important;
}
.hero-dots { display: flex; gap: 6px; margin-top: 10px; }
.hero-dot  { width: 8px; height: 8px; border-radius: 50%; }

.badge {
    display: inline-block;
    padding: 7px 22px;
    border-radius: 24px;
    font-weight: 700;
    font-size: 1rem;
    font-family: 'Palatino Linotype', Georgia, serif !important;
    letter-spacing: 0.02em;
}
.b-Notr   { background: linear-gradient(90deg, #b0d0f0, #7eb8e8); color: white; }
.b-Mutlu  { background: linear-gradient(90deg, #f0c0d8, #e888b8); color: white; }
.b-Ofkeli { background: linear-gradient(90deg, #d8a0d8, #b060c0); color: white; }
.b-Uzgun  { background: linear-gradient(90deg, #a0b8e8, #6090d0); color: white; }
.b-Saskin { background: linear-gradient(90deg, #f0d0e8, #e0a8d0); color: #5a2060; }
.b-unk    { background: #eeeeee; color: #888; }

.info-card {
    background: rgba(255, 245, 252, 0.88);
    border: 1.5px solid #e8b4d4;
    border-radius: 16px;
    padding: 18px 22px;
    color: #4a2d5a !important;
    font-size: .93rem;
    margin-bottom: 14px;
    box-shadow: 0 4px 16px rgba(200, 100, 160, 0.08);
    line-height: 1.7;
}

/* ── MİKROFON KARTI ── */
.mic-card {
    background: linear-gradient(135deg, rgba(240,225,255,0.90), rgba(220,235,255,0.90));
    border: 2px solid #c8a0e0;
    border-radius: 18px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 6px 24px rgba(160,100,200,0.13);
}
.mic-card b { color: #6a3070 !important; }
.mic-card p { color: #5a3868 !important; margin: 6px 0 0; font-size: .9rem; line-height: 1.6; }

.rec-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #e8405a;
    margin-right: 6px;
    animation: blink 1s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

.model-info-card {
    background: linear-gradient(135deg, rgba(255,235,248,0.95), rgba(235,225,255,0.95), rgba(220,235,255,0.95));
    border: 2px solid #dab0e0;
    border-radius: 20px;
    padding: 26px 30px;
    margin-top: 28px;
    box-shadow: 0 8px 30px rgba(180,100,200,0.12);
}
.model-info-card h4 {
    background: linear-gradient(90deg, #b040a0, #7050c0, #4080c0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.18rem;
    font-weight: 700;
    margin-bottom: 16px;
    font-family: 'Palatino Linotype', Georgia, serif !important;
    letter-spacing: 0.01em;
}
.model-section {
    background: rgba(255,255,255,0.65);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 12px;
    border-left: 4px solid;
}
.ms-pink  { border-left-color: #e878b8; }
.ms-lila  { border-left-color: #a868d8; }
.ms-blue  { border-left-color: #68a8e8; }
.ms-mixed { border-left-color: #c090d8; }
.model-section b { color: #6a3070 !important; font-size: .95rem; }
.model-section p { color: #5a3868 !important; margin: 4px 0 0; font-size: .88rem; line-height: 1.6; }
.feat-pill {
    display: inline-block;
    background: linear-gradient(90deg, #f0d0e8, #e0c8f5);
    border: 1px solid #dab0e0;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: .82rem;
    font-weight: 600;
    color: #7040a0 !important;
    margin: 2px 3px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BAŞLIK
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <p class="hero-title">🌸 Duygu Sınıflandırma — Faz 3</p>
  <p class="hero-sub">BIL216 · 2025-2026 Bahar · Grup 10 &nbsp;·&nbsp; Weighted Ensemble (HGB + ExtraTrees + SVM) · 1650 Öznitelik</p>
  <div class="hero-dots">
    <div class="hero-dot" style="background:#e8a0c8;"></div>
    <div class="hero-dot" style="background:#c0a0e0;"></div>
    <div class="hero-dot" style="background:#a0b8e8;"></div>
    <div class="hero-dot" style="background:#e8a0c8;"></div>
    <div class="hero-dot" style="background:#c0a0e0;"></div>
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# DUYGU NORMALİZASYONU
# ─────────────────────────────────────────────
DUYGU_HARITASI = {
    'notr':'Nötr','nötr':'Nötr','neutral':'Nötr','neu':'Nötr',
    'mutlu':'Mutlu','happy':'Mutlu','hap':'Mutlu',
    'ofkeli':'Öfkeli','öfkeli':'Öfkeli','angry':'Öfkeli','ang':'Öfkeli','furious':'Öfkeli',
    'uzgun':'Üzgün','üzgün':'Üzgün','sad':'Üzgün',
    'saskin':'Şaşkın','şaşkın':'Şaşkın','surprised':'Şaşkın',
    'surprise':'Şaşkın','sur':'Şaşkın','shocked':'Şaşkın',
    'saskın':'Şaşkın','şaskın':'Şaşkın',
}
EMO_TR = {
    'Nötr':'Nötr 😐','Mutlu':'Mutlu 😊',
    'Öfkeli':'Öfkeli 😠','Üzgün':'Üzgün 😢','Şaşkın':'Şaşkın 😲',
}
EMO_BADGE = {
    'Nötr':'b-Notr','Mutlu':'b-Mutlu',
    'Öfkeli':'b-Ofkeli','Üzgün':'b-Uzgun','Şaşkın':'b-Saskin',
}
EMO_PALETTE = {
    'Nötr':'#a0c4e8','Mutlu':'#f0a8cc',
    'Öfkeli':'#c8a0e0','Üzgün':'#88b0e0','Şaşkın':'#e8c0d8',
}

def normalize_emotion(raw):
    def tr_lower(s):
        return (s.replace('İ','i').replace('I','ı')
                 .replace('Ğ','ğ').replace('Ü','ü')
                 .replace('Ö','ö').replace('Ş','ş')
                 .replace('Ç','ç').lower())
    key = tr_lower(str(raw).strip())
    return DUYGU_HARITASI.get(key, None)

def label_badge(emo):
    cls = EMO_BADGE.get(emo, 'b-unk')
    txt = EMO_TR.get(emo, emo)
    return f'<span class="badge {cls}">{txt}</span>'

def style_ax(ax, bg="#fdf5fa"):
    ax.set_facecolor(bg)
    ax.tick_params(colors="#5a3060", labelsize=8)
    ax.xaxis.label.set_color("#5a3060")
    ax.yaxis.label.set_color("#5a3060")
    ax.title.set_color("#8040a0")
    for sp in ax.spines.values():
        sp.set_edgecolor("#e0b0d8")

# ─────────────────────────────────────────────
# ÖZNİTELİK ÇIKARMA (1650)
# ─────────────────────────────────────────────
def istatistikleri_hesapla(matris):
    return [
        np.mean(matris, axis=1),
        np.std(matris, axis=1),
        np.max(matris, axis=1),
        np.min(matris, axis=1),
        np.nan_to_num(skew(matris, axis=1),     nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(kurtosis(matris, axis=1), nan=0.0, posinf=0.0, neginf=0.0),
    ]

def oznitelik_cikar(X, sample_rate):
    try:
        X, _ = librosa.effects.trim(X, top_db=20)
        X    = librosa.effects.preemphasis(X)
        stft = np.abs(librosa.stft(X))

        mfccs        = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        chroma       = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        mel          = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        contrast     = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
        tonnetz      = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate)
        zcr          = librosa.feature.zero_crossing_rate(X)
        rms          = librosa.feature.rms(y=X)
        delta_mfccs  = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        return np.hstack(
            istatistikleri_hesapla(mfccs)        +
            istatistikleri_hesapla(delta_mfccs)  +
            istatistikleri_hesapla(delta2_mfccs) +
            istatistikleri_hesapla(chroma)       +
            istatistikleri_hesapla(mel)          +
            istatistikleri_hesapla(contrast)     +
            istatistikleri_hesapla(tonnetz)      +
            istatistikleri_hesapla(zcr)          +
            istatistikleri_hesapla(rms)
        )
    except Exception:
        return None

# ─────────────────────────────────────────────
# AUGMENTASYON
# ─────────────────────────────────────────────
def gurultu_ekle(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])

def pitch_shift(data, sr, n_steps):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

def time_stretch(data, rate=1.1):
    return librosa.effects.time_stretch(y=data, rate=rate)

def augmente_et_ve_cikar(ses_listesi, etiketler):
    X_aug, y_aug = [], []
    for (X, sr), etiket in zip(ses_listesi, etiketler):
        for ses in [X, gurultu_ekle(X), pitch_shift(X, sr, 2), time_stretch(X, 1.1)]:
            oz = oznitelik_cikar(ses, sr)
            if oz is not None:
                X_aug.append(oz)
                y_aug.append(etiket)
    return np.array(X_aug), np.array(y_aug)

def oznitelik_cikar_listeden(ses_listesi):
    return np.array([oz for X, sr in ses_listesi
                     if (oz := oznitelik_cikar(X, sr)) is not None])

# ─────────────────────────────────────────────
# VERİ YÜKLEME
# ─────────────────────────────────────────────
def ham_veri_yukle(metadata_yolu, ses_klasoru):
    df = pd.read_csv(metadata_yolu) if metadata_yolu.endswith('.csv') else pd.read_excel(metadata_yolu)
    ses_listesi, etiketler = [], []
    for _, row in df.iterrows():
        dosya_adi  = str(row['Dosya_Adi']).strip()
        ham_duygu  = str(row['Duygu']).strip().lower()
        dosya_yolu = os.path.join(ses_klasoru, dosya_adi)
        if not os.path.exists(dosya_yolu):
            continue
        emo = normalize_emotion(ham_duygu)
        if emo is None:
            continue
        try:
            X, sr = librosa.load(dosya_yolu, res_type='kaiser_fast')
            ses_listesi.append((X, sr))
            etiketler.append(emo)
        except Exception:
            continue
    return ses_listesi, etiketler

# ─────────────────────────────────────────────
# MODEL OLUŞTURMA
# ─────────────────────────────────────────────
def ensemble_olustur():
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=600, max_depth=15,
        l2_regularization=0.1, random_state=42,
    )
    et = ExtraTreesClassifier(
        n_estimators=700, class_weight='balanced',
        max_depth=25, min_samples_split=5, random_state=42,
    )
    svc = SVC(
        kernel='rbf', C=20, gamma='scale',
        probability=True, class_weight='balanced', random_state=42,
    )
    return VotingClassifier(
        estimators=[('hgb', hgb), ('et', et), ('svc', svc)],
        voting='soft',
        weights=[2, 2, 1],
    )

# ─────────────────────────────────────────────
# TAHMİN
# ─────────────────────────────────────────────
def tahmin_et(dosya_yolu_veya_array, sr=None, model=None, scaler=None):
    if model is None:
        if not os.path.exists('duygu_modeli.pkl'):
            return None, None, "Model dosyası (duygu_modeli.pkl) bulunamadı. Önce eğitim yapın."
        try:
            model  = joblib.load('duygu_modeli.pkl')
            scaler = joblib.load('scaler.pkl')
        except Exception as e:
            return None, None, f"Model yüklenemedi: {e}"

    try:
        if sr is not None:
            X, sample_rate = dosya_yolu_veya_array, sr
        else:
            X, sample_rate = librosa.load(dosya_yolu_veya_array, res_type='kaiser_fast')

        oz = oznitelik_cikar(X, sample_rate)
        if oz is None:
            return None, None, "Öznitelik çıkarılamadı."

        oz_scaled   = scaler.transform([oz])
        tahmin      = model.predict(oz_scaled)[0]
        olasiliklar = model.predict_proba(oz_scaled)[0]
        siniflar    = model.classes_
        sonuc       = dict(zip(siniflar, (olasiliklar * 100).round(2)))
        return tahmin, sonuc, None
    except Exception as e:
        return None, None, str(e)

# ─────────────────────────────────────────────
# MİKROFON — Audio Processor
# ─────────────────────────────────────────────
if WEBRTC_OK:
    class MicAudioProcessor(AudioProcessorBase):
        """
        WebRTC ses çerçevelerini biriktiren işlemci.
        Yayın durduğunda birleştirilmiş ses session_state'e yazılır.
        """
        def __init__(self):
            self._frames: list = []
            self._lock = threading.Lock()

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            pcm = frame.to_ndarray().flatten().astype(np.float32)
            # 16-bit PCM ise normalize et
            if pcm.max() > 1.0:
                pcm = pcm / 32768.0
            with self._lock:
                self._frames.append((pcm, frame.sample_rate))
            return frame

        def get_audio(self):
            with self._lock:
                if not self._frames:
                    return None, None
                sr_ref = self._frames[0][1]
                audio  = np.concatenate([f for f, _ in self._frames])
                return audio, sr_ref

        def clear(self):
            with self._lock:
                self._frames.clear()

# ─────────────────────────────────────────────
# GÖRSEL ÇIZIM YARDIMCISI
# ─────────────────────────────────────────────
def goster_dalga_mfcc(y_in, sr_in):
    fig2 = plt.figure(figsize=(14, 4.5))
    fig2.patch.set_facecolor("#fdf5fa")
    gs2  = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.35)

    ax_w = fig2.add_subplot(gs2[0]); style_ax(ax_w)
    t_arr = np.linspace(0, len(y_in) / sr_in, len(y_in))
    ax_w.plot(t_arr, y_in, color="#c878b0", linewidth=0.5)
    ax_w.fill_between(t_arr, y_in, alpha=0.18, color="#e0a0cc")
    ax_w.set_title("Dalga Formu"); ax_w.set_xlabel("Zaman (s)"); ax_w.set_ylabel("Genlik")

    ax_m = fig2.add_subplot(gs2[1]); style_ax(ax_m)
    mfcc_v = librosa.feature.mfcc(y=y_in, sr=sr_in, n_mfcc=40)
    img    = ax_m.imshow(mfcc_v, aspect='auto', origin='lower', cmap='RdPu')
    plt.colorbar(img, ax=ax_m)
    ax_m.set_title("MFCC Isı Haritası (40 katsayı)")
    ax_m.set_xlabel("Çerçeve"); ax_m.set_ylabel("Katsayı")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

def goster_tahmin_sonucu(tahmin, olasiliklar):
    col_t, col_s = st.columns([1, 2])
    with col_t:
        st.markdown("**Tahmin edilen duygu:**")
        st.markdown(label_badge(tahmin), unsafe_allow_html=True)
        st.markdown(f"**Güven: {olasiliklar[tahmin]:.2f}%**")
        st.markdown("<br>", unsafe_allow_html=True)
        for emo, pct in sorted(olasiliklar.items(), key=lambda x: -x[1]):
            renk = EMO_PALETTE.get(emo, '#e0a0cc')
            star = " ⭐" if emo == tahmin else ""
            st.markdown(
                f"<div style='margin-bottom:6px;'>"
                f"<span style='font-size:.85rem;font-weight:600;color:#6a3070;'>"
                f"{EMO_TR.get(emo,emo)}{star}</span>"
                f"<div style='background:#f5e8f5;border-radius:8px;height:14px;overflow:hidden;margin-top:3px;'>"
                f"<div style='background:{renk};width:{pct:.1f}%;height:100%;border-radius:8px;"
                f"transition:width .4s ease;'></div></div>"
                f"<span style='font-size:.78rem;color:#9060a0;'>{pct:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col_s:
        fig_p, ax_p = plt.subplots(figsize=(7, 3.5))
        fig_p.patch.set_facecolor("#fdf5fa"); style_ax(ax_p)
        siniflar_p  = list(olasiliklar.keys())
        degerler_p  = list(olasiliklar.values())
        bar_renkler = [
            EMO_PALETTE.get(s, '#e0a0cc') if s == tahmin else '#f0d8ea'
            for s in siniflar_p
        ]
        brs = ax_p.bar(
            [EMO_TR.get(s, s) for s in siniflar_p],
            degerler_p, color=bar_renkler, edgecolor='white', width=0.5,
        )
        for b, v in zip(brs, degerler_p):
            ax_p.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                      f"{v:.1f}%", ha='center', va='bottom',
                      color='#5a3060', fontsize=9, fontweight='bold')
        ax_p.set_ylabel("Olasılık (%)"); ax_p.set_ylim(0, 115)
        ax_p.set_title("Ensemble Tahmin Olasılıkları")
        plt.xticks(rotation=15); plt.tight_layout()
        st.pyplot(fig_p)
        plt.close(fig_p)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    metadata_yolu = st.text_input("📋 metadata.xlsx / .csv yolu", value="metadata.xlsx")
    ses_klasoru   = st.text_input("📁 Ses dosyaları klasörü",      value="Ses_Dosyalari")
    st.divider()
    st.markdown("### 💾 Kayıtlı Model")
    model_mevcut = os.path.exists("duygu_modeli.pkl") and os.path.exists("scaler.pkl")
    if model_mevcut:
        st.success("✅ duygu_modeli.pkl mevcut")
    else:
        st.warning("⚠️ Model yok — Eğitim sekmesinden eğitin")
    st.divider()
    test_size = st.slider("Test oranı (%)", 10, 40, 20, 5)
    st.divider()

    wav_files = glob.glob(os.path.join(ses_klasoru, "**", "*.wav"), recursive=True)
    wav_map   = {os.path.basename(f): f for f in wav_files}
    wav_names = sorted(wav_map.keys())
    if wav_names:
        st.success(f"✅ {len(wav_names)} ses dosyası bulundu")
    else:
        st.info("ℹ️ Ses klasöründe .wav bulunamadı")

# ─────────────────────────────────────────────
# SEKMELER
# ─────────────────────────────────────────────
tab0, tab1, tab2, tab3 = st.tabs([
    "🌸 Veri Seti",
    "✨ Eğitim",
    "🎤 Tahmin",
    "📊 Sonuçlar",
])

# ══════════════════════════════════════════════
# TAB 0 — VERİ SETİ
# ══════════════════════════════════════════════
with tab0:
    st.markdown("### 🌸 Veri Seti Analizi")
    if not os.path.exists(metadata_yolu):
        st.error(f"❌ Metadata bulunamadı: `{metadata_yolu}`")
        st.stop()

    df_raw = (pd.read_excel(metadata_yolu)
              if metadata_yolu.endswith(('.xlsx', '.xls'))
              else pd.read_csv(metadata_yolu))
    df_raw.columns = [c.strip() for c in df_raw.columns]
    df = df_raw.copy()
    df['Duygu_N'] = df['Duygu'].apply(normalize_emotion)
    df_valid = df[df['Duygu_N'].notna()].copy()
    st.session_state['df_valid'] = df_valid

    unknown = df[df['Duygu_N'].isna()]['Duygu'].unique()
    if len(unknown):
        st.warning(f"⚠️ Tanınmayan duygu değerleri: {list(unknown)}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📁 Toplam Kayıt",  len(df))
    c2.metric("✅ Geçerli Kayıt", len(df_valid))
    c3.metric("🎭 Duygu Sınıfı",  df_valid['Duygu_N'].nunique())
    c4.metric("🎵 Ses Dosyası",   len(wav_names))

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🎭 Duygu Dağılımı")
        emo_counts = df_valid['Duygu_N'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#fdf5fa"); style_ax(ax)
        colors = [EMO_PALETTE.get(e, '#e0b0d8') for e in emo_counts.index]
        bars = ax.bar([EMO_TR.get(e, e) for e in emo_counts.index],
                      emo_counts.values, color=colors, edgecolor='white', width=0.55)
        for b, v in zip(bars, emo_counts.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, str(v),
                    ha='center', va='bottom', color='#5a3060', fontsize=9, fontweight='bold')
        ax.set_ylabel("Kayıt Sayısı"); plt.xticks(rotation=15); plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
    with col_b:
        st.markdown("#### 📋 Kayıt Tablosu")
        show_df = df_valid[['Dosya_Adi', 'Duygu_N']].copy()
        show_df.columns = ['Dosya Adı', 'Duygu']
        st.dataframe(show_df, use_container_width=True, height=300)

# ══════════════════════════════════════════════
# TAB 1 — EĞİTİM
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### ✨ Model Eğitimi")

    st.markdown("""
    <div class="info-card">
    <b>🌸 Faz 3 Model Mimarisi:</b><br><br>
    🔵 <b>1650 Öznitelik:</b> MFCC×40 + Delta + Delta² · Chroma · Mel Spectrogram · Spectral Contrast · Tonnetz · ZCR · RMS — her grup için 6 istatistik<br>
    🌷 <b>Augmentasyon:</b> Gürültü ekleme · Pitch shift (±2 adım) · Time stretch (×1.1) → eğitim seti ×4 büyür<br>
    💜 <b>SMOTE:</b> Sınıf dengeleme<br>
    💙 <b>Weighted Soft Voting Ensemble:</b><br>
    &nbsp;&nbsp;&nbsp;• HistGradientBoosting (lr=0.08, iter=600, depth=15) — ağırlık: 2<br>
    &nbsp;&nbsp;&nbsp;• ExtraTrees (700 ağaç, depth=25, balanced) — ağırlık: 2<br>
    &nbsp;&nbsp;&nbsp;• SVM RBF (C=20, scale) — ağırlık: 1
    </div>
    """, unsafe_allow_html=True)

    if st.button("✨ Eğitimi Başlat", type="primary"):
        if not os.path.exists(metadata_yolu):
            st.error(f"❌ Metadata bulunamadı: `{metadata_yolu}`"); st.stop()
        if not wav_names:
            st.error("❌ Ses dosyası bulunamadı."); st.stop()

        prog = st.progress(0, "Ses dosyaları yükleniyor...")
        ses_listesi, etiketler = ham_veri_yukle(metadata_yolu, ses_klasoru)
        etiketler = np.array(etiketler)
        prog.progress(20, f"✅ {len(ses_listesi)} dosya yüklendi")

        if len(ses_listesi) < 10:
            st.error("❌ Yeterli örnek yok."); st.stop()

        indisler = np.arange(len(ses_listesi))
        idx_train, idx_test = train_test_split(
            indisler, test_size=test_size / 100, random_state=42, stratify=etiketler,
        )
        train_ses   = [ses_listesi[i] for i in idx_train]
        test_ses    = [ses_listesi[i] for i in idx_test]
        y_train_ham = etiketler[idx_train]
        y_test      = etiketler[idx_test]

        prog.progress(30, "Augmentasyon uygulanıyor (×4)...")
        X_train, y_train = augmente_et_ve_cikar(train_ses, y_train_ham)

        prog.progress(55, "Test seti öznitelikleri çıkarılıyor...")
        X_test = oznitelik_cikar_listeden(test_ses)

        st.info(f"Train: {X_train.shape[0]} örnek | Test: {X_test.shape[0]} örnek | Öznitelik: {X_train.shape[1]}")

        prog.progress(65, "SMOTE uygulanıyor...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        prog.progress(70, "StandardScaler uygulanıyor...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled  = scaler.transform(X_test)

        prog.progress(75, "Ensemble eğitiliyor...")
        model = ensemble_olustur()
        model.fit(X_train_scaled, y_train_res)
        prog.progress(95, "Değerlendiriliyor...")

        y_pred   = model.predict(X_test_scaled)
        dogruluk = accuracy_score(y_test, y_pred)
        prog.progress(100, "✅ Tamamlandı!")

        joblib.dump(model,  'duygu_modeli.pkl')
        joblib.dump(scaler, 'scaler.pkl')

        st.session_state.update({
            'model': model, 'scaler': scaler,
            'y_test': y_test, 'y_pred': y_pred,
            'dogruluk': dogruluk,
            'siniflar': model.classes_,
        })

        st.balloons()
        st.success(f"✅ Model eğitildi! Test Doğruluğu: **{dogruluk*100:.2f}%**")

        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 Test Doğruluğu", f"{dogruluk*100:.2f}%")
        c2.metric("📐 Öznitelik",      X_train.shape[1])
        c3.metric("🔢 Train Örnek",    X_train_res.shape[0])

        st.markdown("#### 📋 Sınıflandırma Raporu")
        cr = classification_report(y_test, y_pred, output_dict=True)
        rows = [{"Duygu": EMO_TR.get(c, c),
                 "Precision": f"{cr.get(c,{}).get('precision',0)*100:.1f}%",
                 "Recall":    f"{cr.get(c,{}).get('recall',0)*100:.1f}%",
                 "F1-Score":  f"{cr.get(c,{}).get('f1-score',0)*100:.1f}%",
                 "Destek":    int(cr.get(c,{}).get('support',0))}
                for c in model.classes_]
        st.table(pd.DataFrame(rows))

# ══════════════════════════════════════════════
# TAB 2 — TAHMİN
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 🎤 Duygu Analizi")

    model_yuklu = 'model'  in st.session_state
    model_disk  = os.path.exists('duygu_modeli.pkl') and os.path.exists('scaler.pkl')

    if not model_yuklu and not model_disk:
        st.warning("⚠️ Model bulunamadı. Lütfen önce **Eğitim** sekmesinden modeli eğitin.")
        st.stop()

    try:
        pipe_model  = st.session_state.get('model',  joblib.load('duygu_modeli.pkl') if model_disk else None)
        pipe_scaler = st.session_state.get('scaler', joblib.load('scaler.pkl')       if model_disk else None)
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}"); st.stop()

    st.markdown("""
    <div class="info-card">
    Ses kaynağı olarak üç seçenek mevcuttur:<br>
    🎙️ <b>Mikrofon ile kayıt</b> — tarayıcı üzerinden canlı ses kaydı yapın<br>
    ⬆️ <b>Dosya yükle</b> — WAV, MP3, OGG veya FLAC dosyası seçin<br>
    📂 <b>Dataset'ten seç</b> — eğitim veri setindeki dosyalardan birini seçin<br><br>
    Model, sesi analiz ederek <b>Nötr · Mutlu · Öfkeli · Üzgün · Şaşkın</b> sınıflarından birini tahmin eder.
    </div>
    """, unsafe_allow_html=True)

    # Kaynak seçimi — mikrofon seçeneği eklendi
    kaynak = st.radio(
        "Kaynak seçin",
        ["🎙️ Mikrofon ile kaydet", "⬆️ Dosya yükle", "📂 Dataset'ten seç"],
        horizontal=True,
    )

    # Kaynak değiştiğinde önceki sonuçları temizle
    if st.session_state.get('_kaynak_onceki') != kaynak:
        for k in ['son_tahmin', 'son_olas', 'p3_y', 'p3_sr', 'p3_fn']:
            st.session_state.pop(k, None)
    st.session_state['_kaynak_onceki'] = kaynak

    # ════════════════════════════════
    # 🎙️  MİKROFON MODU
    # ════════════════════════════════
    if kaynak == "🎙️ Mikrofon ile kaydet":

        if not WEBRTC_OK:
            st.error(
                "❌ **streamlit-webrtc** kurulu değil.\n\n"
                "Terminalde şunu çalıştırın:\n```\npip install streamlit-webrtc av\n```\n"
                "Ardından uygulamayı yeniden başlatın."
            )
        else:
            st.markdown("""
            <div class="mic-card">
            <b>🎙️ Mikrofon Kaydı — Nasıl Kullanılır?</b>
            <p>
            1. <b>START</b> butonuna tıklayın → tarayıcı mikrofon izni isteyecek, izin verin<br>
            2. Duygunuzu yansıtan bir şeyler söyleyin (en az 2-3 saniye)<br>
            3. <b>STOP</b> butonuna tıklayın → kayıt tamamlanır<br>
            4. <b>🔬 Kayıt Analizi Yap</b> butonuna tıklayın → model tahmini başlar
            </p>
            </div>
            """, unsafe_allow_html=True)

            # WebRTC streamer — sadece ses modu
            ctx = webrtc_streamer(
                key="mic_recorder",
                mode=WebRtcMode.SENDONLY,
                audio_processor_factory=MicAudioProcessor,
                media_stream_constraints={"audio": True, "video": False},
                async_processing=True,
            )

            # Kayıt durumu göstergesi
            if ctx.state.playing:
                st.markdown(
                    '<span class="rec-dot"></span>'
                    '<span style="color:#e8405a;font-weight:700;font-size:.95rem;">Kayıt yapılıyor...</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span style="color:#9060a0;font-size:.9rem;">⏹️ Kayıt durduruldu veya henüz başlatılmadı</span>',
                    unsafe_allow_html=True,
                )

            # Analiz butonu
            if st.button("🔬 Kayıt Analizi Yap", type="primary", key="btn_mic"):
                if ctx.audio_processor is None:
                    st.warning("⚠️ Önce START butonuna basıp kayıt yapın, ardından STOP'a basın.")
                else:
                    audio_data, audio_sr = ctx.audio_processor.get_audio()
                    if audio_data is None or len(audio_data) < 100:
                        st.warning("⚠️ Yeterli ses kaydedilemedi. Lütfen daha uzun konuşun.")
                    else:
                        # Ses verisini 22050 Hz'e yeniden örnekle (librosa standardı)
                        if audio_sr != 22050:
                            audio_data = librosa.resample(audio_data, orig_sr=audio_sr, target_sr=22050)
                            audio_sr   = 22050

                        st.session_state.update({
                            'p3_y': audio_data, 'p3_sr': audio_sr, 'p3_fn': '__mic__',
                        })
                        with st.spinner("1650 öznitelik çıkarılıyor ve ensemble tahmin yapılıyor..."):
                            t_res, o_res, h_res = tahmin_et(
                                audio_data, sr=audio_sr,
                                model=pipe_model, scaler=pipe_scaler,
                            )
                        if h_res:
                            st.error(f"Hata: {h_res}")
                        else:
                            st.session_state.update({'son_tahmin': t_res, 'son_olas': o_res})
                        ctx.audio_processor.clear()

        _render_gorsel = st.session_state.get('p3_fn') == '__mic__' and st.session_state.get('p3_y') is not None

    # ════════════════════════════════
    # ⬆️  DOSYA YÜKLE MODU
    # ════════════════════════════════
    elif kaynak == "⬆️ Dosya yükle":
        uploaded = st.file_uploader(
            "Ses dosyası yükleyin (.wav, .mp3, .ogg, .flac)",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
        )
        if uploaded:
            if st.session_state.get('p3_fn') != uploaded.name:
                with tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(uploaded.name)[1], delete=False
                ) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                try:
                    y_loaded, sr_loaded = librosa.load(tmp_path, res_type='kaiser_fast')
                    st.session_state.update({
                        'p3_y': y_loaded, 'p3_sr': sr_loaded,
                        'p3_fn': uploaded.name,
                        'son_tahmin': None, 'son_olas': None,
                    })
                except Exception as e:
                    st.error(f"Dosya yüklenemedi: {e}")
                finally:
                    os.unlink(tmp_path)

            st.audio(uploaded)

            if st.button("🔬 Duygu Analizi Yap", type="primary", key="btn_upload"):
                y_btn  = st.session_state.get('p3_y')
                sr_btn = st.session_state.get('p3_sr')
                if y_btn is not None:
                    with st.spinner("1650 öznitelik çıkarılıyor ve ensemble tahmin yapılıyor..."):
                        t_res, o_res, h_res = tahmin_et(
                            y_btn, sr=sr_btn, model=pipe_model, scaler=pipe_scaler,
                        )
                    if h_res:
                        st.error(f"Hata: {h_res}")
                    else:
                        st.session_state.update({'son_tahmin': t_res, 'son_olas': o_res})

        _render_gorsel = (
            st.session_state.get('p3_y') is not None
            and st.session_state.get('p3_fn') not in (None, '__mic__')
        )

    # ════════════════════════════════
    # 📂  DATASET'TEN SEÇ MODU
    # ════════════════════════════════
    else:
        if not wav_names:
            st.warning("Dataset klasöründe .wav bulunamadı."); st.stop()

        sel = st.selectbox("Ses dosyası", wav_names, format_func=lambda x: f"🎵 {x}")

        if st.button("🔬 Duygu Analizi Yap", type="primary", key="btn_dataset"):
            with st.spinner("Ses yükleniyor ve analiz ediliyor..."):
                y_loaded, sr_loaded = librosa.load(wav_map[sel], res_type='kaiser_fast')
                st.session_state.update({
                    'p3_y': y_loaded, 'p3_sr': sr_loaded, 'p3_fn': sel,
                })
                t_res, o_res, h_res = tahmin_et(
                    y_loaded, sr=sr_loaded, model=pipe_model, scaler=pipe_scaler,
                )
                if h_res:
                    st.error(f"Hata: {h_res}")
                else:
                    st.session_state.update({'son_tahmin': t_res, 'son_olas': o_res})

        if st.session_state.get('p3_fn') and st.session_state['p3_fn'] in wav_map:
            st.audio(wav_map[st.session_state['p3_fn']])

        _render_gorsel = st.session_state.get('p3_y') is not None

    # ════════════════════════════════════════════
    # GÖRSEL + TAHMİN SONUCU (tüm modlar ortak)
    # ════════════════════════════════════════════
    if _render_gorsel and st.session_state.get('p3_y') is not None:
        y_in  = st.session_state['p3_y']
        sr_in = st.session_state['p3_sr']

        st.divider()
        st.markdown("#### 🌊 Dalga Formu & MFCC")
        goster_dalga_mfcc(y_in, sr_in)

        tahmin      = st.session_state.get('son_tahmin')
        olasiliklar = st.session_state.get('son_olas')

        if tahmin is not None and olasiliklar is not None:
            st.divider()
            st.markdown("#### 🎯 Duygu Analizi Sonucu")
            goster_tahmin_sonucu(tahmin, olasiliklar)

# ══════════════════════════════════════════════
# TAB 3 — SONUÇLAR
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Model Başarı Analizi")

    if 'dogruluk' not in st.session_state:
        st.info("ℹ️ Henüz eğitim yapılmadı. Eğitim sekmesinden modeli eğitin."); st.stop()

    dogruluk_v = st.session_state['dogruluk']
    y_test_s   = st.session_state['y_test']
    y_pred_s   = st.session_state['y_pred']
    siniflar_s = st.session_state['siniflar']

    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Test Doğruluğu",   f"{dogruluk_v*100:.2f}%")
    c2.metric("📐 Öznitelik Sayısı", "1650")
    c3.metric("🤖 Model",            "Ensemble (HGB+ET+SVM)")

    st.divider()
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("#### 🔲 Karışıklık Matrisi")
        cm     = confusion_matrix(y_test_s, y_pred_s, labels=siniflar_s)
        labels = [EMO_TR.get(c, c).split()[0] for c in siniflar_s]
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.5))
        fig_cm.patch.set_facecolor("#fdf5fa"); style_ax(ax_cm)
        im = ax_cm.imshow(cm, cmap="RdPu")
        ax_cm.set_xticks(range(len(labels))); ax_cm.set_yticks(range(len(labels)))
        ax_cm.set_xticklabels(labels, color="#5a3060", fontsize=9, fontweight='bold', rotation=20)
        ax_cm.set_yticklabels(labels, color="#5a3060", fontsize=9, fontweight='bold')
        ax_cm.set_xlabel("Tahmin"); ax_cm.set_ylabel("Gerçek")
        ax_cm.set_title(f"Confusion Matrix | Acc: {dogruluk_v*100:.2f}%", fontsize=10, fontweight='bold')
        for i in range(len(labels)):
            for j in range(len(labels)):
                c_txt = "white" if cm[i, j] > cm.max() * 0.5 else "#5a3060"
                ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center',
                           color=c_txt, fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax_cm); plt.tight_layout()
        st.pyplot(fig_cm); plt.close(fig_cm)

    with col_g2:
        st.markdown("#### 📋 Sınıf Bazlı F1 Skorları")
        cr = classification_report(y_test_s, y_pred_s, labels=siniflar_s, output_dict=True)
        f1_vals   = [cr.get(c, {}).get('f1-score', 0) * 100 for c in siniflar_s]
        fig_f1, ax_f1 = plt.subplots(figsize=(5.5, 4.5))
        fig_f1.patch.set_facecolor("#fdf5fa"); style_ax(ax_f1)
        colors_f1 = [EMO_PALETTE.get(c, '#e0a0cc') for c in siniflar_s]
        bars_f1   = ax_f1.bar(labels, f1_vals, color=colors_f1, edgecolor='white', width=0.55)
        for b, v in zip(bars_f1, f1_vals):
            ax_f1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                       f"{v:.1f}%", ha='center', va='bottom',
                       color='#5a3060', fontsize=10, fontweight='bold')
        ax_f1.set_ylabel("F1-Score (%)"); ax_f1.set_ylim(0, 115)
        ax_f1.set_title("Sınıf Bazlı F1 Skorları", fontsize=10, fontweight='bold')
        plt.xticks(rotation=15); plt.tight_layout()
        st.pyplot(fig_f1); plt.close(fig_f1)

    st.divider()
    st.markdown("#### 📋 Detaylı Sınıflandırma Raporu")
    rows = [{"Duygu": EMO_TR.get(c, c),
             "Precision": f"{cr.get(c,{}).get('precision',0)*100:.1f}%",
             "Recall":    f"{cr.get(c,{}).get('recall',0)*100:.1f}%",
             "F1-Score":  f"{cr.get(c,{}).get('f1-score',0)*100:.1f}%",
             "Destek":    int(cr.get(c,{}).get('support',0))}
            for c in siniflar_s]
    st.table(pd.DataFrame(rows))

    csv_data = pd.DataFrame({
        "Gerçek": y_test_s, "Tahmin": y_pred_s, "Doğru": y_test_s == y_pred_s,
    }).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Sonuçları CSV İndir", data=csv_data,
        file_name="Grup10_Faz3_Sonuclar.csv", mime="text/csv",
    )

    st.markdown("""
    <div class="model-info-card">
      <h4>🔬 Kullanılan Model ve Öznitelikler Hakkında</h4>

      <div class="model-section ms-pink">
        <b>🤖 Model: Weighted Soft Voting Ensemble</b>
        <p>
          Üç farklı sınıflandırıcının tahminleri olasılık ortalaması (soft voting) yöntemiyle birleştirilir.
          Her modele farklı ağırlık atanarak güçlü modellerin katkısı artırılır.
        </p>
      </div>

      <div class="model-section ms-lila">
        <b>💜 Alt Modeller ve Parametreler</b>
        <p>
          <b>HistGradientBoostingClassifier (ağırlık: 2):</b> Öğrenme hızı 0.08, 600 iterasyon, maksimum derinlik 15, L2 düzenleştirme 0.1.
          Büyük veri setlerinde hızlı ve yüksek başarılı gradient boosting algoritması.<br><br>
          <b>ExtraTreesClassifier (ağırlık: 2):</b> 700 ağaç, maksimum derinlik 25, balanced sınıf ağırlığı.
          Rastgele bölünme noktaları seçerek aşırı öğrenmeyi azaltır ve çeşitlilik sağlar.<br><br>
          <b>SVM — RBF Kernel (ağırlık: 1):</b> C=20, gamma='scale', balanced sınıf ağırlığı.
          Yüksek boyutlu öznitelik uzayında sınıflar arası maksimum marjin bulmayı hedefler.
        </p>
      </div>

      <div class="model-section ms-blue">
        <b>🎵 Öznitelikler: 1650 Boyutlu Vektör</b>
        <p>
          Her ses dosyasından 9 farklı öznitelik grubu çıkarılır;
          her grup için <i>ortalama, standart sapma, maksimum, minimum, çarpıklık ve basıklık</i>
          olmak üzere 6 istatistik hesaplanır.
        </p>
        <p>
          <span class="feat-pill">MFCC ×40 → 240</span>
          <span class="feat-pill">Delta MFCC ×40 → 240</span>
          <span class="feat-pill">Delta² MFCC ×40 → 240</span>
          <span class="feat-pill">Chroma ×12 → 72</span>
          <span class="feat-pill">Mel Spectrogram ×128 → 768</span>
          <span class="feat-pill">Spectral Contrast ×7 → 42</span>
          <span class="feat-pill">Tonnetz ×6 → 36</span>
          <span class="feat-pill">ZCR ×1 → 6</span>
          <span class="feat-pill">RMS ×1 → 6</span>
        </p>
        <p><b>Toplam: 240 + 240 + 240 + 72 + 768 + 42 + 36 + 6 + 6 = 1650 öznitelik</b></p>
      </div>

      <div class="model-section ms-mixed">
        <b>⚗️ Ön İşleme & Dengeleme</b>
        <p>
          <b>Augmentasyon (×4):</b> Her eğitim örneği orijinal + gürültü ekleme + pitch shift (±2 yarı ton) + time stretch (×1.1) ile 4'e çoğaltılır.<br>
          <b>SMOTE:</b> Sentetik örnekleme ile sınıf dengesizliği giderilir.<br>
          <b>StandardScaler:</b> Tüm öznitelikler sıfır ortalama ve birim varyansa normalize edilir.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)
