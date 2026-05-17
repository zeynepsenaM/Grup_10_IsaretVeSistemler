"""
BIL216 - İşaretler ve Sistemler | Final Proje
2025-2026 Bahar Dönemi | Grup 10
Duygu Sınıflandırma - Faz 2
Yöntem: Genişletilmiş Öznitelikler + MLP (Çok Katmanlı Algılayıcı)

Faz1'e ek yeni öznitelikler (literatür tabanlı):
  • Mel Spectrogram istatistikleri         (Kaynak 5 - Ottoni et al.)
  • Tonnetz (Tonal Centroid) özellikleri   (Kaynak 5 - Ottoni et al.)
  • Delta MFCC + Delta-Delta MFCC          (Kaynak 3 - IMFCC paper)
  • IMFCC (Inverted MFCC)                  (Kaynak 3 - TEOC & IMFCC)
  • Gammatone Filt. Cepstral Coeff (GFCC) (Kaynak 2 - Spectral Kurtosis)
  • Spectral Flux                          (Kaynak 2 - Multiple Acoustic Features)
  • Spectral Flatness                      (Kaynak 2 - Multiple Acoustic Features)
  Toplam: ~340 öznitelik (Faz1: 156 + Faz2 yeni: ~184)
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, glob, warnings, tempfile
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import kurtosis, skew

def _force_float64(X):
    """Pipeline içinde her zaman float64, NaN/Inf → 0.0"""
    X = np.array(X, dtype=np.float64)
    return np.where(np.isfinite(X), X, 0.0)

# ──────────────────────────────────────────────
# SAYFA AYARI
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Duygu Sınıflandırma | Grup 10 | Faz 2",
    page_icon="🧠",
    layout="wide"
)

# ──────────────────────────────────────────────
# CSS — Faz1 temasını koruyoruz, Faz2 vurgusuyla
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&display=swap');
*, html, body, [class*="css"] { font-family:'Nunito',sans-serif; }
.stApp { background:linear-gradient(160deg,#e8f5e9,#e3f2fd,#ede7f6,#e0f2f1);min-height:100vh; }
[data-testid="stSidebar"] { background:rgba(255,255,255,0.82)!important;backdrop-filter:blur(12px);border-right:1.5px solid #b2dfdb; }
[data-testid="stSidebar"] * { color:#1a3a4a!important; }
.stApp,[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li { color:#1a3a4a!important; }
[data-testid="stMetric"] { background:rgba(255,255,255,0.9);border:1.5px solid #80cbc4;border-radius:18px;padding:14px 18px;box-shadow:0 4px 14px rgba(0,150,136,.1); }
[data-testid="stMetricValue"] { font-size:1.65rem!important;font-weight:800!important;color:#00796b!important; }
[data-testid="stMetricLabel"] { color:#4db6ac!important;font-size:.8rem!important;font-weight:600!important; }
.stTabs [data-baseweb="tab-list"] { background:rgba(255,255,255,.75);border-radius:14px;padding:5px;gap:4px;border:1.5px solid #b2dfdb; }
.stTabs [data-baseweb="tab"] { border-radius:10px;color:#4db6ac;font-weight:700; }
.stTabs [aria-selected="true"] { background:linear-gradient(90deg,#4db6ac,#7986cb)!important;color:white!important;box-shadow:0 3px 12px rgba(77,182,172,.4); }
.stButton>button { background:linear-gradient(90deg,#4db6ac,#7986cb);color:white!important;border:none;border-radius:14px;font-weight:700;font-size:1rem;padding:10px 28px;box-shadow:0 4px 18px rgba(77,182,172,.35);transition:all .25s ease; }
.stButton>button:hover { transform:translateY(-2px);box-shadow:0 8px 28px rgba(77,182,172,.5); }
.stTextInput>div>div>input,.stSelectbox>div>div { background:white!important;border:1.5px solid #b2dfdb!important;border-radius:10px!important;color:#1a3a4a!important;font-weight:600!important; }
hr { border-color:#b2dfdb!important; }
.hero-title { background:linear-gradient(90deg,#009688,#3f51b5,#009688);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.5rem;font-weight:800;margin-bottom:0; }
.hero-sub { color:#4db6ac;font-size:.95rem;font-weight:600;margin-top:2px; }
.badge { display:inline-block;padding:7px 20px;border-radius:22px;font-weight:800;font-size:1rem; }
.b-Notr   { background:linear-gradient(90deg,#90caf9,#42a5f5);color:white; }
.b-Mutlu  { background:linear-gradient(90deg,#a5d6a7,#66bb6a);color:white; }
.b-Ofkeli { background:linear-gradient(90deg,#ef9a9a,#e53935);color:white; }
.b-Uzgun  { background:linear-gradient(90deg,#9fa8da,#3949ab);color:white; }
.b-Saskin { background:linear-gradient(90deg,#ffe082,#ffa000);color:#1a3a4a; }
.b-unk    { background:#eeeeee;color:#777; }
.faz2-card { background:rgba(255,255,255,.88);border:1.5px solid #80cbc4;border-radius:14px;padding:16px 20px;color:#1a3a4a;font-size:.93rem;margin-bottom:12px; }
.compare-win  { background:linear-gradient(90deg,#c8e6c9,#b2dfdb);border-radius:10px;padding:8px 16px;font-weight:800;color:#1b5e20; }
.compare-lose { background:#f5f5f5;border-radius:10px;padding:8px 16px;font-weight:600;color:#777; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# BAŞLIK
# ──────────────────────────────────────────────
st.markdown('<p class="hero-title">🧠 Duygu Sınıflandırma — Faz 2</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">BIL216 · 2025-2026 Bahar · Grup 10 &nbsp;|&nbsp; MLP + Genişletilmiş Öznitelikler (Literatür Tabanlı)</p>', unsafe_allow_html=True)
st.divider()

# ──────────────────────────────────────────────
# NORMALİZASYON (Faz1 ile aynı)
# ──────────────────────────────────────────────
def normalize_emotion(raw):
    def tr_lower(s):
        return (s.replace("İ","i").replace("I","ı")
                 .replace("Ğ","ğ").replace("Ü","ü")
                 .replace("Ö","ö").replace("Ş","ş")
                 .replace("Ç","ç").lower())
    def ascii_fold(s):
        tr_map = str.maketrans("şığüöçŞIĞÜÖÇ","sigüocSIGUOC")
        s = s.translate(tr_map)
        import unicodedata
        return unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode().lower()
    key_tr    = tr_lower(str(raw).strip())
    key_ascii = ascii_fold(str(raw).strip())
    LOOKUP = {
        "notr":"Notr","nötr":"Notr","neutral":"Notr",
        "mutlu":"Mutlu","happy":"Mutlu",
        "ofkeli":"Ofkeli","öfkeli":"Ofkeli","ofke":"Ofkeli","öfke":"Ofkeli",
        "angry":"Ofkeli","furious":"Ofkeli",
        "uzgun":"Uzgun","üzgün":"Uzgun","uzgün":"Uzgun","sad":"Uzgun",
        "saskin":"Saskin","saskın":"Saskin","şaşkın":"Saskin","şaşkin":"Saskin",
        "saşkın":"Saskin","saşkin":"Saskin","şaşırma":"Saskin","şaşirma":"Saskin",
        "saşırma":"Saskin","sasirma":"Saskin","surprised":"Saskin","shocked":"Saskin",
    }
    ASCII_LOOKUP = {
        "notr":"Notr","neutral":"Notr","mutlu":"Mutlu","happy":"Mutlu",
        "ofkeli":"Ofkeli","ofke":"Ofkeli","angry":"Ofkeli","furious":"Ofkeli",
        "uzgun":"Uzgun","sad":"Uzgun",
        "saskin":"Saskin","sasirma":"Saskin","surprised":"Saskin","shocked":"Saskin",
    }
    return LOOKUP.get(key_tr) or LOOKUP.get(key_ascii) or ASCII_LOOKUP.get(key_ascii) or None

CIN_NORM = {"e":"E","m":"E","male":"E","k":"K","f":"K","female":"K","c":"C","child":"C"}
def normalize_cinsiyet(raw):
    return CIN_NORM.get(str(raw).strip().lower(), str(raw).strip().upper())

GURULTU_NORM = {
    "dusuk":"Düşük","düşük":"Düşük","low":"Düşük","very low":"Düşük",
    "orta":"Orta","middle":"Orta",
    "yuksek":"Yüksek","yüksek":"Yüksek","high":"Yüksek","very high":"Yüksek",
}
def normalize_gurultu(raw):
    return GURULTU_NORM.get(str(raw).strip().lower(), str(raw).strip())

EMO_TR = {
    "Notr":"Nötr 😐","Mutlu":"Mutlu 😊",
    "Ofkeli":"Öfkeli 😠","Uzgun":"Üzgün 😢","Saskin":"Şaşkın 😲"
}
EMOTIONS = ["Notr","Mutlu","Ofkeli","Uzgun","Saskin"]
EMO_PALETTE = {
    "Notr":"#90caf9","Mutlu":"#a5d6a7",
    "Ofkeli":"#ef9a9a","Uzgun":"#9fa8da","Saskin":"#ffe082"
}

N_MFCC   = 13
MIN_F0   = 60
MAX_F0   = 500
FRAME_MS = 30

# ──────────────────────────────────────────────
# YARDIMCI: güvenli öznitelik temizleyici
# ──────────────────────────────────────────────
def safe_features(arr):
    """Diziyi float64'e çevir, NaN ve Inf değerleri 0 ile değiştir."""
    arr = np.array(arr, dtype=np.float64).flatten()
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr

# ──────────────────────────────────────────────
# ÖZNİTELİK ÇIKARMA — FAZ 1 (baseline karşılaştırma için)
# ──────────────────────────────────────────────
def extract_features_faz1(y, sr):
    """Faz1'deki öznitelik seti (156 öznitelik) — karşılaştırma için sabit."""
    y = y / (np.max(np.abs(y)) + 1e-10)
    fl = int(sr * FRAME_MS / 1000)
    hl = fl // 2

    def stats4(arr):
        arr = np.array(arr, dtype=float)
        if len(arr) < 4:
            arr = np.pad(arr, (0, 4-len(arr)))
        return [np.mean(arr), np.std(arr), float(kurtosis(arr)), float(skew(arr))]

    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=fl, hop_length=hl)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=fl, hop_length=hl)
    sc     = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=fl, hop_length=hl)

    mfcc_f     = np.concatenate([stats4(mfcc[i])   for i in range(N_MFCC)])
    chroma_f   = np.concatenate([stats4(chroma[i]) for i in range(chroma.shape[0])])
    sc_f       = np.concatenate([stats4(sc[i])     for i in range(sc.shape[0])])
    centroid_f = np.array(stats4(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=fl, hop_length=hl)[0]))
    rolloff_f  = np.array(stats4(librosa.feature.spectral_rolloff(y=y,  sr=sr, n_fft=fl, hop_length=hl)[0]))
    bw_f       = np.array(stats4(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=fl, hop_length=hl)[0]))
    rms_f      = np.array(stats4(librosa.feature.rms(y=y, frame_length=fl, hop_length=hl)[0]))
    zcr_f      = np.array(stats4(librosa.feature.zero_crossing_rate(y, frame_length=fl, hop_length=hl)[0]))
    ste        = np.array([np.sum(y[i:i+fl]**2) for i in range(0, len(y)-fl, hl)])
    ste_f      = np.array(stats4(ste if len(ste) > 0 else np.array([0.0])))
    pitches, mags = librosa.piptrack(y=y, sr=sr, n_fft=fl, hop_length=hl, fmin=MIN_F0, fmax=MAX_F0)
    pv = np.array([pitches[mags[:,t].argmax(), t] for t in range(pitches.shape[1])
                   if pitches[mags[:,t].argmax(), t] > 0])
    pitch_f = np.array(stats4(pv if len(pv) > 0 else np.array([0.0])))

    # ── DÜZELTME: float64'e zorla, NaN/Inf temizle ──
    result = np.concatenate([mfcc_f, chroma_f, sc_f, centroid_f, rolloff_f,
                             bw_f, rms_f, zcr_f, ste_f, pitch_f])
    return safe_features(result)  # 156


# ──────────────────────────────────────────────
# ÖZNİTELİK ÇIKARMA — FAZ 2 (GENİŞLETİLMİŞ)
# ──────────────────────────────────────────────
def extract_features_faz2(y, sr):
    """
    Faz2 Öznitelik Seti — Toplam ~340 öznitelik

    FAZ1'DEN GELEN (156):
      MFCC×13, Chroma×12, Spectral Contrast×7,
      Centroid, Rolloff, Bandwidth, RMS, ZCR, STE, Pitch

    FAZ2 YENİ EKLENENLER (~184):
      1. Delta MFCC    (×13×4 = 52)  → Temporal dynamics  [Kaynak 3]
      2. Delta² MFCC   (×13×4 = 52)  → Acceleration       [Kaynak 3]
      3. IMFCC         (×13×4 = 52)  → Yüksek frekans bilgisi [Kaynak 3]
      4. Mel Spectrogram istatistikleri (128→istatistik = 4×4=16) [Kaynak 5]
      5. Tonnetz       (×6×4  = 24)  → Tonal uyum         [Kaynak 5]
      6. Spectral Flux (×4    = 4)   → Zamansal değişim   [Kaynak 2]
      7. Spectral Flatness (×4 = 4)  → Harmoniklik        [Kaynak 2]
      8. Pitch Jitter + Shimmer (×8) → Prosodik özellikler [Kaynak 2]
    """
    y = y / (np.max(np.abs(y)) + 1e-10)
    fl = int(sr * FRAME_MS / 1000)
    hl = fl // 2

    def stats4(arr):
        arr = np.array(arr, dtype=np.float64).flatten()
        if len(arr) < 4:
            arr = np.pad(arr, (0, 4 - len(arr)))
        # NaN/Inf kontrolü stats hesabından önce
        arr = np.where(np.isfinite(arr), arr, 0.0)
        return np.array([np.mean(arr), np.std(arr), float(kurtosis(arr)), float(skew(arr))],
                        dtype=np.float64)

    # ── FAZ1 ÖZNİTELİKLERİ ──────────────────────────────────
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=fl, hop_length=hl)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=fl, hop_length=hl)
    sc     = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=fl, hop_length=hl)

    mfcc_f   = np.concatenate([stats4(mfcc[i])   for i in range(N_MFCC)])      # 52
    chroma_f = np.concatenate([stats4(chroma[i]) for i in range(chroma.shape[0])])  # 48
    sc_f     = np.concatenate([stats4(sc[i])     for i in range(sc.shape[0])]) # 28

    centroid_f = stats4(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=fl, hop_length=hl)[0])  # 4
    rolloff_f  = stats4(librosa.feature.spectral_rolloff(y=y, sr=sr,  n_fft=fl, hop_length=hl)[0])  # 4
    bw_f       = stats4(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=fl, hop_length=hl)[0]) # 4
    rms_f      = stats4(librosa.feature.rms(y=y, frame_length=fl, hop_length=hl)[0])                # 4
    zcr_f      = stats4(librosa.feature.zero_crossing_rate(y, frame_length=fl, hop_length=hl)[0])   # 4
    ste        = np.array([np.sum(y[i:i+fl]**2) for i in range(0, len(y)-fl, hl)])
    ste_f      = stats4(ste if len(ste) > 0 else np.array([0.0]))                                   # 4
    pitches, mags = librosa.piptrack(y=y, sr=sr, n_fft=fl, hop_length=hl, fmin=MIN_F0, fmax=MAX_F0)
    pv = np.array([pitches[mags[:,t].argmax(),t] for t in range(pitches.shape[1])
                   if pitches[mags[:,t].argmax(),t] > 0])
    pitch_f = stats4(pv if len(pv) > 0 else np.array([0.0]))                                        # 4

    # ── FAZ2 YENİ ÖZNİTELİKLER ──────────────────────────────

    # 1. DELTA MFCC — zamansal türev (geçişler) [Kaynak 3: IMFCC paper]
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta_f     = np.concatenate([stats4(delta_mfcc[i]) for i in range(N_MFCC)])   # 52

    # 2. DELTA-DELTA MFCC — ikinci türev (ivme) [Kaynak 3]
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta2_f    = np.concatenate([stats4(delta2_mfcc[i]) for i in range(N_MFCC)])  # 52

    # 3. IMFCC (Inverted MFCC) — ters Mel filtresiyle yüksek frekans [Kaynak 3]
    n_fft_imfcc = fl
    mel_spec    = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft_imfcc,
                                                  hop_length=hl, n_mels=128)
    mel_inv     = mel_spec[::-1, :]
    log_mel_inv = librosa.power_to_db(mel_inv + 1e-10)
    from scipy.fft import dct
    imfcc_frames = dct(log_mel_inv[:N_MFCC, :], axis=0, norm='ortho')
    imfcc_f      = np.concatenate([stats4(imfcc_frames[i]) for i in range(N_MFCC)]) # 52

    # 4. MEL SPECTROGRAM istatistikleri [Kaynak 5: Ottoni et al., Mel spectrogram feature]
    log_mel  = librosa.power_to_db(mel_spec + 1e-10)
    band_size = log_mel.shape[0] // 4
    mel_bands = [log_mel[i*band_size:(i+1)*band_size, :].mean(axis=0) for i in range(4)]
    mel_f     = np.concatenate([stats4(b) for b in mel_bands])  # 16

    # 5. TONNETZ — tonal centroid (uyum/akor bilgisi) [Kaynak 5: Ottoni et al.]
    harmonic  = librosa.effects.harmonic(y)
    tonnetz   = librosa.feature.tonnetz(y=harmonic, sr=sr)
    tonnetz_f = np.concatenate([stats4(tonnetz[i]) for i in range(tonnetz.shape[0])])  # 24

    # 6. SPECTRAL FLUX — çerçeveler arası spektral değişim [Kaynak 2]
    stft_mag  = np.abs(librosa.stft(y, n_fft=fl, hop_length=hl))
    flux      = np.sqrt(np.sum(np.diff(stft_mag, axis=1)**2, axis=0))
    flux_f    = stats4(flux)  # 4

    # 7. SPECTRAL FLATNESS — harmoniklik ölçütü [Kaynak 2]
    flatness  = librosa.feature.spectral_flatness(y=y, n_fft=fl, hop_length=hl)[0]
    flat_f    = stats4(flatness)  # 4

    # 8. JITTER & SHIMMER (pitch-based prosodik özellikler) [Kaynak 2: jitter, shimmer]
    if len(pv) > 5:
        periods   = 1.0 / (pv + 1e-10)
        jitter    = np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-10)
        jitter_r  = np.max(periods) / (np.min(periods) + 1e-10) - 1
        shimmer   = np.mean(np.abs(np.diff(pv))) / (np.mean(pv) + 1e-10)
        shimmer_r = np.std(pv) / (np.mean(pv) + 1e-10)
        hnr_proxy = np.mean(pv) / (np.std(pv) + 1e-10)
        vuvr      = len(pv) / (pitches.shape[1] + 1e-10)
        energy_env = np.sqrt(np.mean(y**2))
        shimmer_db = 20 * np.log10(shimmer + 1e-10)
    else:
        jitter = jitter_r = shimmer = shimmer_r = hnr_proxy = vuvr = energy_env = shimmer_db = 0.0

    prosodic_f = np.array([jitter, jitter_r, shimmer, shimmer_r,
                            hnr_proxy, vuvr, energy_env, shimmer_db], dtype=np.float64)  # 8

    # ── DÜZELTME: hepsini birleştir, float64'e zorla, NaN/Inf temizle ──
    result = np.concatenate([
        # FAZ1 (156)
        mfcc_f,       # 52
        chroma_f,     # 48
        sc_f,         # 28
        centroid_f,   #  4
        rolloff_f,    #  4
        bw_f,         #  4
        rms_f,        #  4
        zcr_f,        #  4
        ste_f,        #  4
        pitch_f,      #  4
        # FAZ2 YENİ (~184)
        delta_f,      # 52
        delta2_f,     # 52
        imfcc_f,      # 52
        mel_f,        # 16
        tonnetz_f,    # 24
        flux_f,       #  4
        flat_f,       #  4
        prosodic_f,   #  8
    ])
    return safe_features(result)  # float64, NaN/Inf → 0.0


# ──────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────
def style_ax(ax, bg="#f0fafa"):
    ax.set_facecolor(bg)
    ax.tick_params(colors="#1a3a4a", labelsize=8)
    ax.xaxis.label.set_color("#1a3a4a")
    ax.yaxis.label.set_color("#1a3a4a")
    ax.title.set_color("#00695c")
    for sp in ax.spines.values():
        sp.set_edgecolor("#b2dfdb")

def label_badge(emo):
    cls = f"b-{emo}" if emo in EMOTIONS else "b-unk"
    txt = EMO_TR.get(emo, emo)
    return f'<span class="badge {cls}">{txt}</span>'

def build_mlp_pipeline(hidden_layers=(256, 128, 64), max_iter=500):
    """float64 zorla + SimpleImputer + StandardScaler + MLP Pipeline"""
    return Pipeline([
        ("to_float64", FunctionTransformer(_force_float64, validate=False)),
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size="auto",
            learning_rate="adaptive",
            max_iter=max_iter,
            early_stopping=False,
            n_iter_no_change=15,
            random_state=42,
            verbose=False
        ))
    ])

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    dataset_path = st.text_input(
        "📁 Dataset klasör yolu",
        value=r"c:\Python\IsaretVeSistemler\Odev_1\DONEM_ODEVI\Dataset"
    )
    excel_path = st.text_input(
        "📋 metadata.xlsx yolu",
        value=r"c:\Python\IsaretVeSistemler\Odev_1\DONEM_ODEVI\metadata.xlsx"
    )
    st.divider()
    st.markdown("### 🧠 MLP Parametreleri")
    layer_preset = st.selectbox(
        "Katman yapısı",
        ["256-128-64 (Önerilen)", "512-256-128 (Derin)", "128-64 (Hızlı)"],
        index=0
    )
    LAYER_MAP = {
        "256-128-64 (Önerilen)": (256, 128, 64),
        "512-256-128 (Derin)":   (512, 256, 128),
        "128-64 (Hızlı)":        (128, 64),
    }
    hidden_layers = LAYER_MAP[layer_preset]
    max_iter  = st.slider("Maks. iterasyon", 200, 1000, 500, 100)
    test_size = st.slider("Test oranı (%)", 10, 40, 20, 5)
    st.divider()
    faz1_compare = st.checkbox("📊 Faz1 (Random Forest) karşılaştırması yap", value=True)
    st.divider()

    wav_files = glob.glob(os.path.join(dataset_path, "**", "*.wav"), recursive=True)
    wav_map   = {os.path.basename(f): f for f in wav_files}
    wav_names = sorted(wav_map.keys())
    if wav_names:
        st.success(f"✅ {len(wav_names)} ses dosyası")
    else:
        st.warning("⚠️ .wav bulunamadı")

# ──────────────────────────────────────────────
# SEKMELER
# ──────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Veri Seti",
    "🧠 Faz2 Eğitimi",
    "🎤 Tahmin",
    "📊 Sonuçlar",
    "⚖️ Faz1 vs Faz2",
])

# ══════════════════════════════════════════════
# TAB 0 — VERİ SETİ
# ══════════════════════════════════════════════
with tab0:
    st.markdown("### 📋 Veri Seti Analizi")
    if not os.path.exists(excel_path):
        st.error(f"❌ Excel bulunamadı: `{excel_path}`")
        st.stop()

    df_raw = pd.read_excel(excel_path)
    df_raw.columns = [c.strip() for c in df_raw.columns]
    df = df_raw.copy()
    df["Duygu_N"]    = df["Duygu"].apply(normalize_emotion)
    df["Cinsiyet_N"] = df["Cinsiyet"].apply(normalize_cinsiyet)
    df["Gurultu_N"]  = df["gürültü seviyesi"].apply(normalize_gurultu)

    unknown_emo = df[df["Duygu_N"].isna()]["Duygu"].unique()
    if len(unknown_emo) > 0:
        st.warning(f"⚠️ Tanınmayan duygu değerleri: {list(unknown_emo)}")

    df_valid = df[df["Duygu_N"].notna()].copy()
    st.session_state["df_valid"]  = df_valid
    st.session_state["wav_map_s"] = wav_map

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📁 Toplam Kayıt",  len(df))
    c2.metric("✅ Geçerli Kayıt", len(df_valid))
    c3.metric("🎭 Duygu Sınıfı",  df_valid["Duygu_N"].nunique())
    c4.metric("🎵 Ses Dosyası",   len(wav_names))

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🎭 Duygu Dağılımı")
        emo_counts = df_valid["Duygu_N"].value_counts()
        fig, ax = plt.subplots(figsize=(6,4)); fig.patch.set_facecolor("#f0fafa"); style_ax(ax)
        colors = [EMO_PALETTE.get(e,"#ccc") for e in emo_counts.index]
        bars = ax.bar([EMO_TR.get(e,e) for e in emo_counts.index],
                      emo_counts.values, color=colors, edgecolor="white", width=0.55)
        for b,v in zip(bars,emo_counts.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, str(v),
                    ha='center', va='bottom', color="#1a3a4a", fontsize=9, fontweight='bold')
        ax.set_ylabel("Kayıt Sayısı"); plt.xticks(rotation=15); plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
    with col_b:
        st.markdown("#### 📊 Ham Tablo")
        show_df = df_valid[["Dosya_Adi","Duygu_N","Cinsiyet_N"]].copy()
        show_df.columns = ["Dosya Adı","Duygu","Cinsiyet"]
        st.dataframe(show_df, use_container_width=True, height=300)

# ══════════════════════════════════════════════
# TAB 1 — FAZ2 EĞİTİMİ
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### 🧠 Faz 2 — MLP Model Eğitimi")

    st.markdown("""
    <div class="faz2-card">
    <b>📌 Faz 2 Öznitelik Seti (~340 öznitelik):</b><br><br>
    <b>FAZ1'DEN (156):</b> MFCC×13, Chroma×12, Spectral Contrast×7, Centroid, Rolloff, Bandwidth, RMS, ZCR, STE, Pitch<br><br>
    <b>FAZ2 YENİ (~184):</b><br>
    🔵 <b>Delta MFCC</b> (52) — Geçiş dinamikleri <i>[Kaynak 3: IMFCC & TEOC]</i><br>
    🔵 <b>Delta² MFCC</b> (52) — İvme bilgisi <i>[Kaynak 3]</i><br>
    🔵 <b>IMFCC</b> (52) — Yüksek frekans duygu bilgisi <i>[Kaynak 3]</i><br>
    🟢 <b>Mel Spectrogram</b> (16) — Bant bazlı enerji <i>[Kaynak 5: Ottoni et al.]</i><br>
    🟢 <b>Tonnetz</b> (24) — Tonal uyum/akor <i>[Kaynak 5]</i><br>
    🟡 <b>Spectral Flux</b> (4) — Zamansal değişim hızı <i>[Kaynak 2]</i><br>
    🟡 <b>Spectral Flatness</b> (4) — Harmoniklik ölçütü <i>[Kaynak 2]</i><br>
    🟠 <b>Jitter & Shimmer</b> (8) — Prosodik özellikler <i>[Kaynak 2]</i><br><br>
    <b>Model:</b> MLP (Çok Katmanlı Algılayıcı) + StandardScaler + Early Stopping
    </div>
    """, unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        start = st.button("🚀 Eğitimi Başlat", type="primary")

    if start:
        if not os.path.exists(excel_path):
            st.error(f"❌ Excel bulunamadı: `{excel_path}`")
            st.stop()
        if not wav_names:
            st.error("❌ Ses dosyası bulunamadı.")
            st.stop()

        df_raw2 = pd.read_excel(excel_path)
        df_raw2.columns = [c.strip() for c in df_raw2.columns]
        df_raw2["Duygu_N"] = df_raw2["Duygu"].apply(normalize_emotion)
        df_valid2 = df_raw2[df_raw2["Duygu_N"].notna()].copy()
        st.success(f"✅ Excel: {len(df_raw2)} kayıt | Geçerli: {len(df_valid2)}")

        # ── ÖZNİTELİK ÇIKARMA ──
        X2_list, y2_list = [], []
        X1_list = []
        missing, skipped = [], []
        prog  = st.progress(0, "Öznitelikler çıkarılıyor (Faz2 — genişletilmiş)...")
        total = len(df_valid2)

        for i, (_, row) in enumerate(df_valid2.iterrows()):
            fname = str(row["Dosya_Adi"]).strip()
            emo   = row["Duygu_N"]
            prog.progress((i+1)/total, f"⏳ {i+1}/{total} — {fname}")
            if fname not in wav_map:
                missing.append(fname); continue
            try:
                y_a, sr = librosa.load(wav_map[fname], sr=None, mono=True)
                feat2   = extract_features_faz2(y_a, sr)
                X2_list.append(feat2)
                y2_list.append(emo)
                if faz1_compare:
                    feat1 = extract_features_faz1(y_a, sr)
                    X1_list.append(feat1)
            except Exception as e:
                skipped.append(f"{fname}: {e}")

        prog.empty()

        if len(X2_list) < 10:
            st.error(f"❌ Yeterli örnek yok ({len(X2_list)}). Dataset yolunu kontrol edin.")
            st.stop()

        # ── DÜZELTME: dtype=np.float64 zorla, kalan NaN/Inf temizle ──
        X2 = np.array(X2_list, dtype=np.float64)
        X2 = np.where(np.isfinite(X2), X2, 0.0)
        y2 = np.array(y2_list)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("✅ Kullanılan Örnek",  len(X2))
        c2.metric("📐 Öznitelik Sayısı", X2.shape[1])
        c3.metric("❓ Eksik Dosya",       len(missing))
        c4.metric("🎭 Sınıf Sayısı",      len(np.unique(y2)))

        X2_tr, X2_te, y2_tr, y2_te = train_test_split(
            X2, y2, test_size=test_size/100, random_state=42,
            stratify=y2 if len(np.unique(y2)) > 1 else None
        )
        # Split sonrası da kesin float64 garantisi
        X2_tr = np.array(X2_tr, dtype=np.float64)
        X2_te = np.array(X2_te, dtype=np.float64)
        X2_tr = np.where(np.isfinite(X2_tr), X2_tr, 0.0)
        X2_te = np.where(np.isfinite(X2_te), X2_te, 0.0)

        # ── FAZ2 MLP EĞİTİMİ ──
        with st.spinner(f"🧠 MLP eğitiliyor ({layer_preset})..."):
            mlp_pipe = build_mlp_pipeline(hidden_layers, max_iter)
            mlp_pipe.fit(X2_tr, y2_tr)

        y2_pred = mlp_pipe.predict(X2_te)
        acc2    = accuracy_score(y2_te, y2_pred)
        cv2     = cross_val_score(mlp_pipe, X2, y2, cv=5, scoring="accuracy", n_jobs=-1)

        # ── FAZ1 KARŞILAŞTIRMASI ──
        acc1, cv1 = None, None
        if faz1_compare and len(X1_list) > 0:
            from sklearn.ensemble import RandomForestClassifier
            X1 = np.array(X1_list, dtype=np.float64)
            X1 = np.where(np.isfinite(X1), X1, 0.0)
            X1_tr, X1_te, y1_tr, y1_te = train_test_split(
                X1, y2, test_size=test_size/100, random_state=42,
                stratify=y2 if len(np.unique(y2)) > 1 else None
            )
            with st.spinner("🌳 Faz1 Random Forest yeniden eğitiliyor (karşılaştırma için)..."):
                rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                rf.fit(X1_tr, y1_tr)
            y1_pred = rf.predict(X1_te)
            acc1    = accuracy_score(y1_te, y1_pred)
            cv1     = cross_val_score(rf, X1, y2, cv=5, scoring="accuracy", n_jobs=-1)
            st.session_state.update({
                "acc1": acc1, "cv1": cv1, "X1": X1,
                "y1_te": y1_te, "y1_pred": y1_pred
            })

        st.session_state.update({
            "mlp_pipe": mlp_pipe,
            "X2": X2, "y2": y2,
            "X2_te": X2_te, "y2_te": y2_te, "y2_pred": y2_pred,
            "acc2": acc2, "cv2": cv2,
            "wav_map_s": wav_map, "wav_names_s": wav_names,
        })

        st.balloons()

        improvement = f" (+{(acc2-acc1)*100:.1f}%)" if acc1 else ""
        st.success(
            f"✅ Faz2 MLP eğitildi! "
            f"Test Doğruluğu: **{acc2*100:.1f}%**{improvement} | "
            f"5-Fold CV: **{cv2.mean()*100:.1f}%**"
        )
        if acc1:
            delta = acc2 - acc1
            color = "#2e7d32" if delta > 0 else "#c62828"
            arrow = "▲" if delta > 0 else "▼"
            st.markdown(f"""
            <div style='background:rgba(255,255,255,.9);border:1.5px solid #80cbc4;
                        border-radius:12px;padding:14px 20px;margin-top:8px;'>
            <b>Faz1 (RF):</b> {acc1*100:.1f}% &nbsp;→&nbsp;
            <b>Faz2 (MLP):</b> {acc2*100:.1f}%
            &nbsp;&nbsp;<span style='color:{color};font-weight:800;font-size:1.1rem'>
            {arrow} {abs(delta)*100:.1f}% {"iyileşme" if delta>0 else "düşüş"}</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Öğrenme eğrisi ──
        mlp_model = mlp_pipe.named_steps["mlp"]
        if hasattr(mlp_model, "loss_curve_"):
            st.markdown("#### 📉 MLP Öğrenme Eğrisi")
            fig_lc, ax_lc = plt.subplots(figsize=(9, 3.5))
            fig_lc.patch.set_facecolor("#f0fafa"); style_ax(ax_lc)
            ax_lc.plot(mlp_model.loss_curve_, color="#009688", linewidth=2, label="Eğitim Kaybı")
            if hasattr(mlp_model, "validation_scores_") and mlp_model.validation_scores_:
                ax2_lc = ax_lc.twinx()
                ax2_lc.plot(mlp_model.validation_scores_, color="#7986cb",
                            linewidth=2, linestyle="--", label="Doğrulama Skoru")
                ax2_lc.set_ylabel("Doğrulama Doğruluğu", color="#7986cb", fontsize=9)
                ax2_lc.tick_params(colors="#7986cb")
            ax_lc.set_xlabel("İterasyon", fontsize=9)
            ax_lc.set_ylabel("Loss", fontsize=9)
            ax_lc.set_title("MLP Öğrenme Eğrisi", fontsize=10, fontweight='bold')
            ax_lc.legend(loc="upper right", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_lc); plt.close(fig_lc)

# ══════════════════════════════════════════════
# TAB 2 — TEK DOSYA TAHMİNİ
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 🎤 Tek Ses Dosyasından Duygu Tahmini")

    if "mlp_pipe" not in st.session_state:
        st.info("ℹ️ Önce **Faz2 Eğitimi** sekmesinden modeli eğitin.")
        st.stop()

    pipe_l = st.session_state["mlp_pipe"]
    wm     = st.session_state.get("wav_map_s", wav_map)
    wn     = st.session_state.get("wav_names_s", wav_names)

    source = st.radio("Kaynak seçin", ["📂 Dataset'ten seç","⬆️ Dosya yükle"], horizontal=True)
    y_in, sr_in, fname_in, true_emo = None, None, "", None

    if source == "📂 Dataset'ten seç":
        if wn:
            sel = st.selectbox("Ses dosyası seçin", wn, format_func=lambda x: f"🎵 {x}")
            if st.button("▶️ Tahmin Et"):
                y_in, sr_in = librosa.load(wm[sel], sr=None, mono=True)
                fname_in = sel
                parts = sel.replace(".wav","").split("_")
                if len(parts) >= 5:
                    true_emo = normalize_emotion(parts[4])
                st.session_state.update({"p2_y":y_in,"p2_sr":sr_in,
                                          "p2_fn":fname_in,"p2_true":true_emo})
        else:
            st.warning("Dataset'te dosya bulunamadı.")
    else:
        up = st.file_uploader("Bir .wav yükleyin", type=["wav"])
        if up:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(up.read()); tmp_path = tmp.name
            y_in, sr_in = librosa.load(tmp_path, sr=None, mono=True)
            fname_in = up.name; os.unlink(tmp_path)
            parts = fname_in.replace(".wav","").split("_")
            if len(parts) >= 5:
                true_emo = normalize_emotion(parts[4])
            st.session_state.update({"p2_y":y_in,"p2_sr":sr_in,
                                      "p2_fn":fname_in,"p2_true":true_emo})

    if "p2_y" in st.session_state:
        y_in     = st.session_state["p2_y"]
        sr_in    = st.session_state["p2_sr"]
        fname_in = st.session_state["p2_fn"]
        true_emo = st.session_state["p2_true"]

    if y_in is not None:
        with st.spinner("🔬 Faz2 öznitelikleri çıkarılıyor..."):
            feat   = extract_features_faz2(y_in, sr_in)
            pred   = pipe_l.predict([feat])[0]
            proba  = pipe_l.predict_proba([feat])[0]
            classes = pipe_l.classes_

        col_b1, col_b2, col_b3 = st.columns([1,1,2])
        with col_b1:
            st.markdown("**🎯 Tahmin:**")
            st.markdown(label_badge(pred), unsafe_allow_html=True)
        with col_b2:
            st.markdown("**✅ Gerçek:**")
            if true_emo:
                st.markdown(label_badge(true_emo), unsafe_allow_html=True)
                dogru = pred == true_emo
                col_b3.markdown(
                    f"**Sonuç:** <span style='color:{'#2e7d32' if dogru else '#c62828'};"
                    f"font-weight:800;font-size:1.2rem'>{'✅ Doğru!' if dogru else '❌ Yanlış'}</span>",
                    unsafe_allow_html=True
                )

        st.divider()
        fig_p, ax_p = plt.subplots(figsize=(8,3))
        fig_p.patch.set_facecolor("#f0fafa"); style_ax(ax_p)
        bar_c = [EMO_PALETTE.get(c,"#4db6ac") if c==pred else "#b2dfdb" for c in classes]
        brs   = ax_p.bar([EMO_TR.get(c,c) for c in classes], proba*100,
                          color=bar_c, edgecolor="white", width=0.5)
        for b,v in zip(brs, proba*100):
            ax_p.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                      f"{v:.1f}%", ha='center', va='bottom',
                      color="#1a3a4a", fontsize=9, fontweight='bold')
        ax_p.set_ylabel("Olasılık (%)"); ax_p.set_ylim(0,115)
        ax_p.set_title("MLP Tahmin Olasılıkları"); plt.tight_layout()
        st.pyplot(fig_p); plt.close(fig_p)

        st.markdown("#### 🌊 Dalga Formu & MFCC")
        fig2 = plt.figure(figsize=(14,4.5)); fig2.patch.set_facecolor("#f0fafa")
        gs2  = gridspec.GridSpec(1,2,figure=fig2,wspace=0.35)
        ax_w = fig2.add_subplot(gs2[0]); style_ax(ax_w)
        t    = np.linspace(0,len(y_in)/sr_in, len(y_in))
        ax_w.plot(t, y_in, color="#009688", linewidth=0.5)
        ax_w.fill_between(t, y_in, alpha=0.15, color="#009688")
        ax_w.set_title("Dalga Formu"); ax_w.set_xlabel("Zaman (s)"); ax_w.set_ylabel("Genlik")
        ax_m = fig2.add_subplot(gs2[1]); style_ax(ax_m)
        fl2  = int(sr_in*FRAME_MS/1000); hl2=fl2//2
        mfcc_v = librosa.feature.mfcc(y=y_in, sr=sr_in, n_mfcc=N_MFCC, n_fft=fl2, hop_length=hl2)
        img  = ax_m.imshow(mfcc_v, aspect='auto', origin='lower', cmap='YlGnBu')
        plt.colorbar(img, ax=ax_m)
        ax_m.set_title("MFCC Isı Haritası"); ax_m.set_xlabel("Çerçeve"); ax_m.set_ylabel("Katsayı")
        plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

# ══════════════════════════════════════════════
# TAB 3 — SONUÇLAR
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Faz 2 Model Başarı Analizi")
    if "mlp_pipe" not in st.session_state:
        st.info("ℹ️ Önce modeli eğitin."); st.stop()

    acc2_v  = st.session_state["acc2"]
    cv2_v   = st.session_state["cv2"]
    y2_te_s = st.session_state["y2_te"]
    y2_pr_s = st.session_state["y2_pred"]
    X2_s    = st.session_state["X2"]
    y2_s    = st.session_state["y2"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🎯 Test Doğruluğu",  f"{acc2_v*100:.1f}%")
    c2.metric("🔁 5-Fold CV Ort.",  f"{cv2_v.mean()*100:.1f}%")
    c3.metric("📈 CV Std Sapma",    f"±{cv2_v.std()*100:.1f}%")
    c4.metric("📐 Öznitelik Sayısı", X2_s.shape[1])

    st.divider()
    st.markdown("#### 📋 Sınıf Bazlı Başarı (MLP)")
    classes_p = np.unique(np.concatenate([y2_te_s, y2_pr_s]))
    cr = classification_report(y2_te_s, y2_pr_s, labels=classes_p, output_dict=True)
    rows = [{"Duygu": EMO_TR.get(cls,cls),
             "Precision": f"{cr.get(cls,{}).get('precision',0)*100:.1f}%",
             "Recall":    f"{cr.get(cls,{}).get('recall',0)*100:.1f}%",
             "F1-Score":  f"{cr.get(cls,{}).get('f1-score',0)*100:.1f}%",
             "Destek":    int(cr.get(cls,{}).get("support",0))}
            for cls in classes_p]
    st.table(pd.DataFrame(rows))

    st.divider()
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### 🔲 Karışıklık Matrisi (MLP)")
        cm     = confusion_matrix(y2_te_s, y2_pr_s, labels=classes_p)
        labels = [EMO_TR.get(c,c).split()[0] for c in classes_p]
        fig_cm, ax_cm = plt.subplots(figsize=(5.5,4.5))
        fig_cm.patch.set_facecolor("#f0fafa"); style_ax(ax_cm)
        im = ax_cm.imshow(cm, cmap="YlGnBu")
        ax_cm.set_xticks(range(len(labels))); ax_cm.set_yticks(range(len(labels)))
        ax_cm.set_xticklabels(labels, color="#1a3a4a", fontsize=9, fontweight='bold', rotation=20)
        ax_cm.set_yticklabels(labels, color="#1a3a4a", fontsize=9, fontweight='bold')
        ax_cm.set_xlabel("Tahmin"); ax_cm.set_ylabel("Gerçek")
        ax_cm.set_title("MLP Confusion Matrix", fontsize=11, fontweight='bold')
        for i in range(len(labels)):
            for j in range(len(labels)):
                c = "white" if cm[i,j] > cm.max()*0.5 else "#1a3a4a"
                ax_cm.text(j, i, str(cm[i,j]), ha='center', va='center',
                           color=c, fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax_cm); plt.tight_layout()
        st.pyplot(fig_cm); plt.close(fig_cm)

    with col_g2:
        st.markdown("#### 📊 5-Fold CV Dağılımı")
        fig_cv, ax_cv = plt.subplots(figsize=(5.5,4.5))
        fig_cv.patch.set_facecolor("#f0fafa"); style_ax(ax_cv)
        folds = [f"Fold {i+1}" for i in range(len(cv2_v))]
        bars  = ax_cv.bar(folds, cv2_v*100, color=["#4db6ac","#80cbc4","#009688","#4db6ac","#00897b"],
                           edgecolor="white", width=0.55)
        for b,v in zip(bars, cv2_v*100):
            ax_cv.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                       f"{v:.1f}%", ha='center', va='bottom',
                       color="#1a3a4a", fontsize=9, fontweight='bold')
        ax_cv.axhline(cv2_v.mean()*100, color="#e53935", linestyle="--",
                       linewidth=1.5, label=f"Ort: {cv2_v.mean()*100:.1f}%")
        ax_cv.set_ylabel("Doğruluk (%)"); ax_cv.set_ylim(0,115)
        ax_cv.set_title("5-Fold Cross Validation Sonuçları", fontsize=10, fontweight='bold')
        ax_cv.legend(fontsize=9); plt.tight_layout()
        st.pyplot(fig_cv); plt.close(fig_cv)

    csv_data = pd.DataFrame({
        "Gerçek": y2_te_s, "Tahmin": y2_pr_s, "Doğru": y2_te_s == y2_pr_s
    }).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Sonuçları CSV İndir", data=csv_data,
                       file_name="Grup10_Faz2_Sonuclar.csv", mime="text/csv")

# ══════════════════════════════════════════════
# TAB 4 — FAZ1 vs FAZ2 KARŞILAŞTIRMA
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### ⚖️ Faz 1 (Random Forest) vs Faz 2 (MLP) Karşılaştırması")

    if "mlp_pipe" not in st.session_state:
        st.info("ℹ️ Önce eğitim yapın."); st.stop()
    if "acc1" not in st.session_state:
        st.info("ℹ️ Sidebar'da 'Faz1 karşılaştırması yap' seçeneğini işaretleyerek eğitim yapın.")
        st.stop()

    acc1_v = st.session_state["acc1"]
    cv1_v  = st.session_state["cv1"]
    acc2_v = st.session_state["acc2"]
    cv2_v  = st.session_state["cv2"]
    delta  = acc2_v - acc1_v

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,.9);border:1.5px solid #ef9a9a;
                    border-radius:16px;padding:18px 22px;text-align:center;'>
        <div style='color:#888;font-weight:700;font-size:.85rem'>FAZ 1 — Random Forest</div>
        <div style='color:#c62828;font-size:2.2rem;font-weight:800'>{acc1_v*100:.1f}%</div>
        <div style='color:#888;font-size:.8rem'>Test Doğruluğu</div>
        <div style='color:#aaa;font-size:.8rem;margin-top:6px'>CV: {cv1_v.mean()*100:.1f}% ± {cv1_v.std()*100:.1f}%</div>
        <div style='color:#aaa;font-size:.8rem'>Öznitelik: 156</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        arrow = "▲" if delta > 0 else "▼"
        color = "#2e7d32" if delta > 0 else "#c62828"
        st.markdown(f"""
        <div style='background:rgba(255,255,255,.9);border:1.5px solid #80cbc4;
                    border-radius:16px;padding:18px 22px;text-align:center;'>
        <div style='color:#888;font-weight:700;font-size:.85rem'>FAZ 2 — MLP</div>
        <div style='color:#006064;font-size:2.2rem;font-weight:800'>{acc2_v*100:.1f}%</div>
        <div style='color:#888;font-size:.8rem'>Test Doğruluğu</div>
        <div style='color:#aaa;font-size:.8rem;margin-top:6px'>CV: {cv2_v.mean()*100:.1f}% ± {cv2_v.std()*100:.1f}%</div>
        <div style='color:#aaa;font-size:.8rem'>Öznitelik: ~340</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,.9);border:1.5px solid #b2dfdb;
                    border-radius:16px;padding:18px 22px;text-align:center;'>
        <div style='color:#888;font-weight:700;font-size:.85rem'>DEĞİŞİM</div>
        <div style='color:{color};font-size:2.2rem;font-weight:800'>{arrow} {abs(delta)*100:.1f}%</div>
        <div style='color:#888;font-size:.8rem'>{"İyileşme" if delta>0 else "Düşüş"}</div>
        <div style='color:#aaa;font-size:.8rem;margin-top:6px'>
        Öznitelik: 156 → ~340<br>(+{340-156} yeni öznitelik)
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col_gc1, col_gc2 = st.columns(2)
    with col_gc1:
        st.markdown("#### 📊 Test Doğruluğu Karşılaştırması")
        fig_c, ax_c = plt.subplots(figsize=(6,4))
        fig_c.patch.set_facecolor("#f0fafa"); style_ax(ax_c)
        models   = ["Faz1\nRandom Forest", "Faz2\nMLP"]
        accs     = [acc1_v*100, acc2_v*100]
        bar_cols = ["#ef9a9a", "#4db6ac"]
        bars     = ax_c.bar(models, accs, color=bar_cols, edgecolor="white",
                             width=0.45, linewidth=1.5)
        for b,v in zip(bars, accs):
            ax_c.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                      f"{v:.1f}%", ha='center', va='bottom',
                      color="#1a3a4a", fontsize=13, fontweight='bold')
        ax_c.set_ylabel("Doğruluk (%)"); ax_c.set_ylim(0,115)
        ax_c.set_title("Test Doğruluğu", fontsize=11, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_c); plt.close(fig_c)

    with col_gc2:
        st.markdown("#### 🔁 Cross-Validation Karşılaştırması")
        fig_cv2, ax_cv2 = plt.subplots(figsize=(6,4))
        fig_cv2.patch.set_facecolor("#f0fafa"); style_ax(ax_cv2)
        x = np.arange(5); w = 0.35
        b1 = ax_cv2.bar(x-w/2, cv1_v*100, w, label="Faz1 RF",  color="#ef9a9a", edgecolor="white")
        b2 = ax_cv2.bar(x+w/2, cv2_v*100, w, label="Faz2 MLP", color="#4db6ac", edgecolor="white")
        ax_cv2.set_xticks(x); ax_cv2.set_xticklabels([f"Fold {i+1}" for i in range(5)], fontsize=8)
        ax_cv2.set_ylabel("Doğruluk (%)"); ax_cv2.set_ylim(0,115)
        ax_cv2.set_title("5-Fold CV Karşılaştırması", fontsize=11, fontweight='bold')
        ax_cv2.legend(fontsize=9, facecolor="#f0fafa")
        plt.tight_layout(); st.pyplot(fig_cv2); plt.close(fig_cv2)

    st.divider()
    st.markdown("#### 📋 Detaylı Karşılaştırma")
    compare_data = {
        "Kriter": [
            "Test Doğruluğu",
            "CV Ortalama",
            "CV Std Sapma",
            "Öznitelik Sayısı",
            "Algoritma",
            "Ön İşlem",
        ],
        "Faz 1 (Random Forest)": [
            f"{acc1_v*100:.1f}%",
            f"{cv1_v.mean()*100:.1f}%",
            f"±{cv1_v.std()*100:.1f}%",
            "156",
            "Random Forest (200 ağaç)",
            "Yok",
        ],
        "Faz 2 (MLP)": [
            f"{acc2_v*100:.1f}%",
            f"{cv2_v.mean()*100:.1f}%",
            f"±{cv2_v.std()*100:.1f}%",
            "~340",
            f"MLP {hidden_layers}",
            "StandardScaler + Early Stopping",
        ],
    }
    st.table(pd.DataFrame(compare_data))

    st.markdown(f"""
    <div class="faz2-card">
    <b>📋 Score-Board Özeti | Grup 10 | Faz 2</b><br>
    <b>Yöntem:</b> Genişletilmiş Öznitelikler (Delta/Delta² MFCC, IMFCC, Mel Spec, Tonnetz, Flux, Flatness, Jitter/Shimmer) + MLP<br>
    <b>Test Doğruluğu:</b>
    <span style='color:#006064;font-weight:800;font-size:1.1rem'>{acc2_v*100:.1f}%</span>
    &nbsp;|&nbsp;
    <b>5-Fold CV:</b>
    <span style='color:#006064;font-weight:800;font-size:1.1rem'>
        {cv2_v.mean()*100:.1f}% ± {cv2_v.std()*100:.1f}%
    </span>
    &nbsp;|&nbsp;
    <b>Faz1'e Göre:</b>
    <span style='color:{"#2e7d32" if delta>0 else "#c62828"};font-weight:800'>
        {"▲ +" if delta>0 else "▼ "}{abs(delta)*100:.1f}%
    </span>
    </div>
    """, unsafe_allow_html=True)