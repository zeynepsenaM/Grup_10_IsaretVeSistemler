"""
BIL216 - İşaretler ve Sistemler | Final Proje
2025-2026 Bahar Dönemi | Grup 10
Duygu Sınıflandırma - Faz 1
Yöntem: MFCC + ZCR + STE + Pitch → Random Forest
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, glob, warnings, tempfile
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ──────────────────────────────────────────────
# SAYFA AYARI
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Duygu Sınıflandırma | Grup 10",
    page_icon="🎭",
    layout="wide"
)

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&display=swap');
*, html, body, [class*="css"] { font-family:'Nunito',sans-serif; }
.stApp { background:linear-gradient(160deg,#fff0f6,#fce4ec,#f3e5f5,#ede7f6);min-height:100vh; }
[data-testid="stSidebar"] { background:rgba(255,255,255,0.78)!important;backdrop-filter:blur(12px);border-right:1.5px solid #f8bbd0; }
[data-testid="stSidebar"] * { color:#6d3b6e!important; }
.stApp,[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li { color:#4a2040!important; }
[data-testid="stMetric"] { background:rgba(255,255,255,0.88);border:1.5px solid #f8bbd0;border-radius:18px;padding:14px 18px;box-shadow:0 4px 14px rgba(233,30,99,.08); }
[data-testid="stMetricValue"] { font-size:1.65rem!important;font-weight:800!important;color:#c2185b!important; }
[data-testid="stMetricLabel"] { color:#ad6f8a!important;font-size:.8rem!important;font-weight:600!important; }
.stTabs [data-baseweb="tab-list"] { background:rgba(255,255,255,.7);border-radius:14px;padding:5px;gap:4px;border:1.5px solid #f8bbd0; }
.stTabs [data-baseweb="tab"] { border-radius:10px;color:#ad6f8a;font-weight:700; }
.stTabs [aria-selected="true"] { background:linear-gradient(90deg,#f48fb1,#ce93d8)!important;color:white!important;box-shadow:0 3px 12px rgba(244,143,177,.4); }
.stButton>button { background:linear-gradient(90deg,#f48fb1,#ce93d8);color:white!important;border:none;border-radius:14px;font-weight:700;font-size:1rem;padding:10px 28px;box-shadow:0 4px 18px rgba(244,143,177,.4);transition:all .25s ease; }
.stButton>button:hover { transform:translateY(-2px);box-shadow:0 8px 28px rgba(244,143,177,.55); }
.stTextInput>div>div>input,.stSelectbox>div>div { background:white!important;border:1.5px solid #f8bbd0!important;border-radius:10px!important;color:#4a2040!important;font-weight:600!important; }
hr { border-color:#f8bbd0!important; }
.hero-title { background:linear-gradient(90deg,#e91e63,#9c27b0,#e91e63);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.5rem;font-weight:800;margin-bottom:0; }
.hero-sub { color:#ad6f8a;font-size:.95rem;font-weight:600;margin-top:2px; }
.badge { display:inline-block;padding:7px 20px;border-radius:22px;font-weight:800;font-size:1rem; }
.b-Notr   { background:linear-gradient(90deg,#90caf9,#42a5f5);color:white; }
.b-Mutlu  { background:linear-gradient(90deg,#a5d6a7,#66bb6a);color:white; }
.b-Ofkeli { background:linear-gradient(90deg,#ef9a9a,#e53935);color:white; }
.b-Uzgun  { background:linear-gradient(90deg,#9fa8da,#3949ab);color:white; }
.b-Saskin { background:linear-gradient(90deg,#ffe082,#ffa000);color:#4a2040; }
.b-unk    { background:#eeeeee;color:#777; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# BAŞLIK
# ──────────────────────────────────────────────
st.markdown('<p class="hero-title">🎭 Duygu Sınıflandırma Sistemi</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">BIL216 · 2025-2026 Bahar · Grup 10 &nbsp;|&nbsp; Final Proje — Faz 1</p>', unsafe_allow_html=True)
st.divider()

# ──────────────────────────────────────────────
# NORMALİZASYON — TR + EN, unicode-safe
# ──────────────────────────────────────────────
def normalize_emotion(raw):
    """
    Türkçe unicode sorunlarını (ş/s, ı/i vb.) aşmak için
    hem orijinal hem ASCII-katlanmış forma bakılır.
    """
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
        # Nötr
        "notr":"Notr","nötr":"Notr","neutral":"Notr",
        # Mutlu
        "mutlu":"Mutlu","happy":"Mutlu",
        # Öfkeli
        "ofkeli":"Ofkeli","öfkeli":"Ofkeli",
        "ofke":"Ofkeli","öfke":"Ofkeli",
        "angry":"Ofkeli","furious":"Ofkeli",
        # Üzgün
        "uzgun":"Uzgun","üzgün":"Uzgun","uzgün":"Uzgun","sad":"Uzgun",
        # Şaşkın — tüm yazılışlar
        "saskin":"Saskin","saskın":"Saskin",
        "şaşkın":"Saskin","şaşkin":"Saskin",
        "saşkın":"Saskin","saşkin":"Saskin",
        "şaşırma":"Saskin","şaşirma":"Saskin",
        "saşırma":"Saskin","sasirma":"Saskin",
        "surprised":"Saskin","shocked":"Saskin",
    }
    ASCII_LOOKUP = {
        "notr":"Notr","neutral":"Notr",
        "mutlu":"Mutlu","happy":"Mutlu",
        "ofkeli":"Ofkeli","ofke":"Ofkeli","angry":"Ofkeli","furious":"Ofkeli",
        "uzgun":"Uzgun","sad":"Uzgun",
        "saskin":"Saskin","sasirma":"Saskin","surprised":"Saskin","shocked":"Saskin",
    }
    return LOOKUP.get(key_tr) or LOOKUP.get(key_ascii) or ASCII_LOOKUP.get(key_ascii) or None

# Cinsiyet normalize
CIN_NORM = {
    "e": "E", "m": "E", "male": "E",
    "k": "K", "f": "K", "female": "K",
    "c": "C", "child": "C",
}
def normalize_cinsiyet(raw):
    return CIN_NORM.get(str(raw).strip().lower(), str(raw).strip().upper())

# Gürültü normalize
GURULTU_NORM = {
    "dusuk": "Düşük", "düşük": "Düşük", "low": "Düşük", "very low": "Düşük",
    "orta": "Orta", "middle": "Orta",
    "yuksek": "Yüksek", "yüksek": "Yüksek", "high": "Yüksek", "very high": "Yüksek",
}
def normalize_gurultu(raw):
    return GURULTU_NORM.get(str(raw).strip().lower(), str(raw).strip())

EMO_TR = {
    "Notr": "Nötr 😐", "Mutlu": "Mutlu 😊",
    "Ofkeli": "Öfkeli 😠", "Uzgun": "Üzgün 😢",
    "Saskin": "Şaşkın 😲"
}
EMOTIONS = ["Notr", "Mutlu", "Ofkeli", "Uzgun", "Saskin"]
EMO_PALETTE = {
    "Notr": "#90caf9", "Mutlu": "#a5d6a7",
    "Ofkeli": "#ef9a9a", "Uzgun": "#9fa8da", "Saskin": "#ffe082"
}

N_MFCC = 13
MIN_F0 = 60
MAX_F0 = 500
FRAME_MS = 30

# ──────────────────────────────────────────────
# ÖZNİTELİK ÇIKARMA
# ──────────────────────────────────────────────
def extract_features(y, sr):
    """
    Öznitelik seti (toplam ~76 öznitelik):
    ZAMAN DÜZLEMİ  : ZCR, STE, Pitch
    FREKANS DÜZLEMİ: MFCC×13, Chroma×12, Spectral Contrast×7,
                     Spectral Centroid, Spectral Rolloff,
                     Spectral Bandwidth, RMS Energy
    İSTATİSTİKSEL  : Her biri için ort + std + kurtosis + skewness
    """
    from scipy.stats import kurtosis, skew

    y = y / (np.max(np.abs(y)) + 1e-10) #genlik normalizasyonu
    fl = int(sr * FRAME_MS / 1000) #30ms pencere
    hl = fl // 2   # %50 örtüşme

    def stats4(arr):
        """Ortalama, Std, Kurtosis, Skewness döndür."""
        return [np.mean(arr), np.std(arr),
                float(kurtosis(arr)) if len(arr) > 3 else 0.0,
                float(skew(arr))     if len(arr) > 3 else 0.0]

    # ── MFCC (13 katsayı × 4 istatistik = 52) ──
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=fl, hop_length=hl)
    mfcc_feats = np.concatenate([stats4(mfcc[i]) for i in range(N_MFCC)])

    # ── Chroma (12 × 4 = 48) ──
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=fl, hop_length=hl)
    chroma_feats = np.concatenate([stats4(chroma[i]) for i in range(chroma.shape[0])])

    # ── Spectral Contrast (7 × 4 = 28) ──
    sc = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=fl, hop_length=hl)
    sc_feats = np.concatenate([stats4(sc[i]) for i in range(sc.shape[0])])

    # ── Spectral Centroid (4) ──
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=fl, hop_length=hl)[0]
    centroid_feats = np.array(stats4(centroid))

    # ── Spectral Rolloff (4) ──
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=fl, hop_length=hl)[0]
    rolloff_feats = np.array(stats4(rolloff))

    # ── Spectral Bandwidth (4) ──
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=fl, hop_length=hl)[0]
    bandwidth_feats = np.array(stats4(bandwidth))

    # ── RMS Enerji (4) ──
    rms = librosa.feature.rms(y=y, frame_length=fl, hop_length=hl)[0]
    rms_feats = np.array(stats4(rms))

    # ── ZCR (4) ──
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=fl, hop_length=hl)[0]
    zcr_feats = np.array(stats4(zcr))

    # ── STE (4) ──
    ste = np.array([np.sum(y[i:i+fl]**2) for i in range(0, len(y)-fl, hl)])
    ste_feats = np.array(stats4(ste))

    # ── Pitch (4) ──
    pitches, mags = librosa.piptrack(y=y, sr=sr, n_fft=fl, hop_length=hl,
                                     fmin=MIN_F0, fmax=MAX_F0)
    pitch_vals = [pitches[mags[:, t].argmax(), t]
                  for t in range(pitches.shape[1])
                  if pitches[mags[:, t].argmax(), t] > 0]
    pv = np.array(pitch_vals) if pitch_vals else np.array([0.0])
    pitch_feats = np.array(stats4(pv))

    return np.concatenate([
        mfcc_feats,       # 52
        chroma_feats,     # 48
        sc_feats,         # 28
        centroid_feats,   #  4
        rolloff_feats,    #  4
        bandwidth_feats,  #  4
        rms_feats,        #  4
        zcr_feats,        #  4
        ste_feats,        #  4
        pitch_feats,      #  4
    ])  # Toplam: 156 öznitelik

def style_ax(ax, bg="#fff5f8"):
    ax.set_facecolor(bg)
    ax.tick_params(colors="#4a2040", labelsize=8)
    ax.xaxis.label.set_color("#4a2040")
    ax.yaxis.label.set_color("#4a2040")
    ax.title.set_color("#6d3b6e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#f8bbd0")

def label_badge(emo):
    cls = f"b-{emo}" if emo in EMOTIONS else "b-unk"
    txt = EMO_TR.get(emo, emo)
    return f'<span class="badge {cls}">{txt}</span>'

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    dataset_path = st.text_input("📁 Dataset klasör yolu", value=r"c:\Python\IsaretVeSistemler\Odev_1\DONEM_ODEVI\Dataset")
    excel_path   = st.text_input("📋 metadata.xlsx yolu",  value=r"c:\Python\IsaretVeSistemler\Odev_1\DONEM_ODEVI\metadata.xlsx")
    st.divider()
    st.markdown("### 🌳 Model Parametreleri")
    n_trees   = st.slider("Ağaç sayısı (n_estimators)", 50, 500, 200, 50)
    test_size = st.slider("Test oranı (%)", 10, 40, 20, 5)
    st.divider()

    wav_files = glob.glob(os.path.join(dataset_path, "**", "*.wav"), recursive=True)
    wav_map   = {os.path.basename(f): f for f in wav_files}
    wav_names = sorted(wav_map.keys())
    if wav_names:
        st.success(f"✅ {len(wav_names)} ses dosyası")
    else:
        st.warning("⚠️ .wav bulunamadı — yolu kontrol edin")

# ──────────────────────────────────────────────
# SEKMELER
# ──────────────────────────────────────────────
tab0, tab1, tab2, tab3 = st.tabs([
    "📋 Veri Seti Özeti",
    "🏋️ Model Eğitimi",
    "🎤 Tek Dosya Tahmini",
    "📊 Sonuçlar & Başarı",
])

# ══════════════════════════════════════════════
# TAB 0 — VERİ SETİ ÖZETİ
# ══════════════════════════════════════════════
with tab0:
    st.markdown("### 📋 Veri Seti Analizi")

    if not os.path.exists(excel_path):
        st.error(f"❌ Excel bulunamadı: `{excel_path}`")
        st.stop()

    df_raw = pd.read_excel(excel_path)
    df_raw.columns = [c.strip() for c in df_raw.columns]

    # Normalize et
    df = df_raw.copy()
    df["Duygu_N"]    = df["Duygu"].apply(normalize_emotion)
    df["Cinsiyet_N"] = df["Cinsiyet"].apply(normalize_cinsiyet)
    df["Gurultu_N"]  = df["gürültü seviyesi"].apply(normalize_gurultu)

    # Etiket tanınmayanları göster
    unknown_emo = df[df["Duygu_N"].isna()]["Duygu"].unique()
    if len(unknown_emo) > 0:
        st.warning(f"⚠️ Tanınmayan duygu değerleri (atlanacak): {list(unknown_emo)}")

    df_valid = df[df["Duygu_N"].notna()].copy()

    # Genel istatistikler
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📁 Toplam Kayıt",    len(df))
    c2.metric("✅ Geçerli Kayıt",   len(df_valid))
    c3.metric("🎭 Duygu Sınıfı",    df_valid["Duygu_N"].nunique())
    # Denek_ID normalize et (D01, 1 → aynı sayılmasın diye grup+denek bazlı say)
    denek_norm = df["Dosya_Adi"].str.extract(r"(G\d+_D\d+)")[0].nunique()
    c4.metric("👥 Toplam Denek", denek_norm)

    st.divider()

    col_g1, col_g2 = st.columns(2)

    # Duygu dağılımı
    with col_g1:
        st.markdown("#### 🎭 Duygu Dağılımı")
        emo_counts = df_valid["Duygu_N"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        fig1.patch.set_facecolor("#fff0f6")
        style_ax(ax1)
        colors = [EMO_PALETTE.get(e, "#ccc") for e in emo_counts.index]
        bars = ax1.bar(
            [EMO_TR.get(e, e) for e in emo_counts.index],
            emo_counts.values, color=colors, edgecolor="white", linewidth=0.7, width=0.55
        )
        for b, v in zip(bars, emo_counts.values):
            ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                     str(v), ha='center', va='bottom', color="#4a2040",
                     fontsize=9, fontweight='bold')
        ax1.set_ylabel("Kayıt Sayısı", fontsize=9)
        ax1.set_title("Duygu Sınıfı Dağılımı", fontsize=10, fontweight='bold')
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig1); plt.close(fig1)

    # Cinsiyet dağılımı
    with col_g2:
        st.markdown("#### 👥 Cinsiyet Dağılımı")
        cin_counts = df_valid["Cinsiyet_N"].value_counts()
        cin_labels = {"E": "Erkek 👨", "K": "Kadın 👩", "C": "Çocuk 🧒"}
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor("#fff0f6")
        style_ax(ax2)
        cin_colors = {"E": "#90caf9", "K": "#f48fb1", "C": "#a5d6a7"}
        bars2 = ax2.bar(
            [cin_labels.get(c, c) for c in cin_counts.index],
            cin_counts.values,
            color=[cin_colors.get(c, "#ccc") for c in cin_counts.index],
            edgecolor="white", linewidth=0.7, width=0.45
        )
        for b, v in zip(bars2, cin_counts.values):
            ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                     str(v), ha='center', va='bottom', color="#4a2040",
                     fontsize=9, fontweight='bold')
        ax2.set_ylabel("Kayıt Sayısı", fontsize=9)
        ax2.set_title("Cinsiyet Dağılımı", fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2); plt.close(fig2)

    st.divider()

    col_g3, col_g4 = st.columns(2)

    # Gürültü dağılımı
    with col_g3:
        st.markdown("#### 🔊 Gürültü Seviyesi Dağılımı")
        gur_counts = df_valid["Gurultu_N"].value_counts()
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        fig3.patch.set_facecolor("#fff0f6")
        style_ax(ax3)
        gur_colors = ["#ce93d8", "#f48fb1", "#ef9a9a"]
        ax3.bar(gur_counts.index, gur_counts.values,
                color=gur_colors[:len(gur_counts)], edgecolor="white", linewidth=0.7, width=0.4)
        for i, (idx, v) in enumerate(gur_counts.items()):
            ax3.text(i, v+0.3, str(v), ha='center', va='bottom',
                     color="#4a2040", fontsize=9, fontweight='bold')
        ax3.set_ylabel("Kayıt Sayısı", fontsize=9)
        ax3.set_title("Gürültü Seviyesi", fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3); plt.close(fig3)

    # Duygu × Cinsiyet ısı haritası
    with col_g4:
        st.markdown("#### 🎭×👥 Duygu × Cinsiyet")
        cross = pd.crosstab(df_valid["Duygu_N"], df_valid["Cinsiyet_N"])
        cross.index = [EMO_TR.get(e, e).split()[0] for e in cross.index]
        cross.columns = [{"E":"Erkek","K":"Kadın","C":"Çocuk"}.get(c,c) for c in cross.columns]
        fig4, ax4 = plt.subplots(figsize=(6, 3.5))
        fig4.patch.set_facecolor("#fff0f6")
        style_ax(ax4, bg="#fff0f6")
        im = ax4.imshow(cross.values, cmap="RdPu", aspect="auto")
        ax4.set_xticks(range(len(cross.columns)))
        ax4.set_yticks(range(len(cross.index)))
        ax4.set_xticklabels(cross.columns, color="#4a2040", fontsize=9)
        ax4.set_yticklabels(cross.index, color="#4a2040", fontsize=9)
        for i in range(len(cross.index)):
            for j in range(len(cross.columns)):
                v = cross.values[i, j]
                c = "white" if v > cross.values.max()*0.5 else "#4a2040"
                ax4.text(j, i, str(v), ha='center', va='center',
                         color=c, fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax4)
        ax4.set_title("Duygu × Cinsiyet Dağılımı", color="#6d3b6e",
                      fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig4); plt.close(fig4)

    st.divider()
    st.markdown("#### 🗂️ Ham Veri Tablosu")
    show_df = df_valid[["Dosya_Adi","Duygu_N","Cinsiyet_N","Yas","Gurultu_N","ORTAM"]].copy()
    show_df.columns = ["Dosya Adı","Duygu","Cinsiyet","Yaş","Gürültü","Ortam"]
    st.dataframe(show_df, use_container_width=True, height=300)

    # Session'a kaydet (eğitim sekmesi için)
    st.session_state["df_valid"]  = df_valid
    st.session_state["wav_map_s"] = wav_map
    st.session_state["wav_names_s"] = wav_names

# ══════════════════════════════════════════════
# TAB 1 — MODEL EĞİTİMİ
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### 🏋️ Model Eğitimi")

    st.markdown("""
    <div style='background:rgba(255,255,255,.85);border:1.5px solid #f8bbd0;
                border-radius:14px;padding:16px 20px;color:#4a2040;font-size:.93rem;'>
    <b>📌 Yöntem:</b> Her ses dosyasından <b>156 öznitelik</b> çıkarılır:<br>
    🎵 <b>Frekans Düzlemi:</b> MFCC×13, Chroma×12, Spectral Contrast×7,
    Spectral Centroid, Rolloff, Bandwidth, RMS<br>
    ⏱️ <b>Zaman Düzlemi:</b> ZCR, STE, Pitch<br>
    📊 <b>İstatistiksel:</b> Her biri için ortalama + std + kurtosis + skewness<br>
    🌳 <b>Model:</b> Random Forest → 5 duygu sınıfı
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    if st.button("🚀 Eğitimi Başlat", type="primary"):
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

        X_list, y_list, missing, skipped = [], [], [], []
        prog  = st.progress(0, "Öznitelikler çıkarılıyor...")
        total = len(df_valid2)

        for i, (_, row) in enumerate(df_valid2.iterrows()):
            fname = str(row["Dosya_Adi"]).strip()
            emo   = row["Duygu_N"]
            prog.progress((i+1)/total, f"⏳ {i+1}/{total} — {fname}")

            if fname not in wav_map:
                missing.append(fname); continue
            try:
                y_a, sr = librosa.load(wav_map[fname], sr=None, mono=True)
                feat    = extract_features(y_a, sr)
                X_list.append(feat)
                y_list.append(emo)
            except Exception:
                skipped.append(fname)

        prog.empty()

        if len(X_list) < 10:
            st.error(f"❌ Yeterli örnek yok ({len(X_list)}). Dataset yolunu kontrol edin.")
            st.stop()

        X = np.array(X_list)
        y = np.array(y_list)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Kullanılan Örnek", len(X))
        c2.metric("❓ Eksik Dosya",      len(missing))
        c3.metric("⚠️ Atlanan",          len(skipped))
        c4.metric("🎭 Sınıf Sayısı",     len(np.unique(y)))

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size/100, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        with st.spinner("🌳 Random Forest eğitiliyor..."):
            clf = RandomForestClassifier(
                n_estimators=n_trees, max_depth=None,
                random_state=42, n_jobs=-1
            )
            clf.fit(X_tr, y_tr)

        y_pred  = clf.predict(X_te)
        acc     = accuracy_score(y_te, y_pred)
        cv_sc   = cross_val_score(clf, X, y, cv=5, scoring="accuracy") # 5 katlı çapraz doğrulama

        st.session_state.update({
            "model": clf, "X": X, "y": y,
            "X_te": X_te, "y_te": y_te, "y_pred": y_pred,
            "acc": acc, "cv": cv_sc,
            "wav_map_s": wav_map, "wav_names_s": wav_names,
        })

        st.balloons()
        st.success(
            f"✅ Model eğitildi! "
            f"Test Doğruluğu: **{acc*100:.1f}%** | "
            f"5-Fold CV: **{cv_sc.mean()*100:.1f}%**"
        )

        # Öznitelik önemi
        feat_names = (
            [f"MFCC{i+1}_{s}" for i in range(N_MFCC) for s in ["ort","std","kurt","skew"]] +
            [f"Chroma{i+1}_{s}" for i in range(12) for s in ["ort","std","kurt","skew"]] +
            [f"SC{i+1}_{s}" for i in range(7) for s in ["ort","std","kurt","skew"]] +
            [f"Centroid_{s}" for s in ["ort","std","kurt","skew"]] +
            [f"Rolloff_{s}" for s in ["ort","std","kurt","skew"]] +
            [f"Bandwidth_{s}" for s in ["ort","std","kurt","skew"]] +
            [f"RMS_{s}" for s in ["ort","std","kurt","skew"]] +
            [f"ZCR_{s}" for s in ["ort","std","kurt","skew"]] +
            [f"STE_{s}" for s in ["ort","std","kurt","skew"]] +
            [f"Pitch_{s}" for s in ["ort","std","kurt","skew"]]
        )
        importances = clf.feature_importances_
        top_idx     = np.argsort(importances)[::-1][:15]

        fig_fi, ax_fi = plt.subplots(figsize=(9, 4))
        fig_fi.patch.set_facecolor("#fff0f6")
        style_ax(ax_fi)
        ax_fi.barh([feat_names[i] for i in top_idx[::-1]],
                   [importances[i] for i in top_idx[::-1]],
                   color="#f48fb1", edgecolor="white", linewidth=0.5)
        ax_fi.set_xlabel("Önem Skoru", fontsize=9)
        ax_fi.set_title("Top 15 Öznitelik Önemi", fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_fi); plt.close(fig_fi)

# ══════════════════════════════════════════════
# TAB 2 — TEK DOSYA TAHMİNİ
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 🎤 Tek Ses Dosyasından Duygu Tahmini")

    if "model" not in st.session_state:
        st.info("ℹ️ Önce **Model Eğitimi** sekmesinden modeli eğitin.")
        st.stop()

    clf_l = st.session_state["model"]
    wm    = st.session_state.get("wav_map_s", wav_map)
    wn    = st.session_state.get("wav_names_s", wav_names)

    source = st.radio("Kaynak seçin",
                      ["📂 Dataset'ten seç", "⬆️ Dosya yükle"],
                      horizontal=True)

    y_in, sr_in, fname_in, true_emo = None, None, "", None

    if source == "📂 Dataset'ten seç":
        if wn:
            sel = st.selectbox("Ses dosyası seçin", wn,
                               format_func=lambda x: f"🎵 {x}")
            if st.button("▶️ Tahmin Et"):
                y_in, sr_in = librosa.load(wm[sel], sr=None, mono=True)
                fname_in    = sel
                parts = sel.replace(".wav","").split("_")
                if len(parts) >= 5:
                    true_emo = normalize_emotion(parts[4])
                st.session_state.update({"p_y":y_in,"p_sr":sr_in,
                                         "p_fn":fname_in,"p_true":true_emo})
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
            st.session_state.update({"p_y":y_in,"p_sr":sr_in,
                                     "p_fn":fname_in,"p_true":true_emo})

    if "p_y" in st.session_state:
        y_in     = st.session_state["p_y"]
        sr_in    = st.session_state["p_sr"]
        fname_in = st.session_state["p_fn"]
        true_emo = st.session_state["p_true"]

    if y_in is not None:
        with st.spinner("🔬 Analiz yapılıyor..."):
            feat  = extract_features(y_in, sr_in)
            pred  = clf_l.predict([feat])[0]
            proba = clf_l.predict_proba([feat])[0]
            classes = clf_l.classes_

        st.divider()
        col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
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
            else:
                st.markdown('<span class="badge b-unk">Bilinmiyor</span>',
                            unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 📊 Sınıf Olasılıkları")
        fig_p, ax_p = plt.subplots(figsize=(8, 3))
        fig_p.patch.set_facecolor("#fff0f6")
        style_ax(ax_p)
        bar_c = [EMO_PALETTE.get(c, "#ce93d8") if c == pred else "#d4b8e0"
                 for c in classes]
        brs = ax_p.bar([EMO_TR.get(c, c) for c in classes],
                       proba*100, color=bar_c, edgecolor="white",
                       linewidth=0.6, width=0.5)
        for b, v in zip(brs, proba*100):
            ax_p.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                      f"{v:.1f}%", ha='center', va='bottom',
                      color="#4a2040", fontsize=9, fontweight='bold')
        ax_p.set_ylabel("Olasılık (%)", fontsize=9)
        ax_p.set_title("Tahmin Olasılıkları", fontsize=10, fontweight='bold')
        ax_p.set_ylim(0, 115)
        plt.tight_layout()
        st.pyplot(fig_p); plt.close(fig_p)

        # Dalga + MFCC
        st.markdown("#### 🌊 Dalga Formu & MFCC Isı Haritası")
        fig2 = plt.figure(figsize=(14, 4.5))
        fig2.patch.set_facecolor("#fff0f6")
        gs2  = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.35)

        ax_w = fig2.add_subplot(gs2[0]); style_ax(ax_w)
        t = np.linspace(0, len(y_in)/sr_in, len(y_in))
        ax_w.plot(t, y_in, color="#e91e63", linewidth=0.5)
        ax_w.fill_between(t, y_in, alpha=0.15, color="#e91e63")
        ax_w.set_title("Dalga Formu", fontsize=10, fontweight='bold')
        ax_w.set_xlabel("Zaman (s)", fontsize=8); ax_w.set_ylabel("Genlik", fontsize=8)

        ax_m = fig2.add_subplot(gs2[1]); style_ax(ax_m, bg="#fff0f6")
        fl = int(sr_in * FRAME_MS / 1000); hl = fl // 2
        mfcc_full = librosa.feature.mfcc(y=y_in, sr=sr_in,
                                          n_mfcc=N_MFCC, n_fft=fl, hop_length=hl)
        img = ax_m.imshow(mfcc_full, aspect='auto', origin='lower',
                          cmap='RdPu', interpolation='nearest')
        plt.colorbar(img, ax=ax_m)
        ax_m.set_title("MFCC Isı Haritası", fontsize=10, fontweight='bold')
        ax_m.set_xlabel("Çerçeve", fontsize=8); ax_m.set_ylabel("MFCC Katsayısı", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2); plt.close(fig2)

# ══════════════════════════════════════════════
# TAB 3 — SONUÇLAR & BAŞARI
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Model Başarı Analizi")

    if "model" not in st.session_state:
        st.info("ℹ️ Önce **Model Eğitimi** sekmesinden modeli eğitin.")
        st.stop()

    acc_v  = st.session_state["acc"]
    cv_v   = st.session_state["cv"]
    y_te_s = st.session_state["y_te"]
    y_pr_s = st.session_state["y_pred"]
    X_s    = st.session_state["X"]
    y_s    = st.session_state["y"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Test Doğruluğu", f"{acc_v*100:.1f}%")
    c2.metric("🔁 5-Fold CV Ort.", f"{cv_v.mean()*100:.1f}%")
    c3.metric("📈 CV Std Sapma",   f"±{cv_v.std()*100:.1f}%")
    c4.metric("📁 Toplam Örnek",   len(X_s))

    st.divider()

    # Sınıflandırma raporu
    st.markdown("#### 📋 Sınıf Bazlı Başarı")
    classes_p = np.unique(np.concatenate([y_te_s, y_pr_s]))
    cr = classification_report(y_te_s, y_pr_s, labels=classes_p, output_dict=True)
    rows = []
    for cls in classes_p:
        r = cr.get(cls, {})
        rows.append({
            "Duygu":     EMO_TR.get(cls, cls),
            "Precision": f"{r.get('precision',0)*100:.1f}%",
            "Recall":    f"{r.get('recall',0)*100:.1f}%",
            "F1-Score":  f"{r.get('f1-score',0)*100:.1f}%",
            "Destek":    int(r.get("support", 0))
        })
    st.table(pd.DataFrame(rows))

    st.divider()
    col_g1, col_g2 = st.columns(2)

    # Confusion Matrix
    with col_g1:
        st.markdown("#### 🔲 Karışıklık Matrisi")
        cm     = confusion_matrix(y_te_s, y_pr_s, labels=classes_p)
        labels = [EMO_TR.get(c,c).split()[0] for c in classes_p]
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.5))
        fig_cm.patch.set_facecolor("#fff0f6"); style_ax(ax_cm)
        im = ax_cm.imshow(cm, cmap="RdPu")
        ax_cm.set_xticks(range(len(labels))); ax_cm.set_yticks(range(len(labels)))
        ax_cm.set_xticklabels(labels, color="#4a2040", fontsize=9,
                               fontweight='bold', rotation=20)
        ax_cm.set_yticklabels(labels, color="#4a2040", fontsize=9, fontweight='bold')
        ax_cm.set_xlabel("Tahmin", color="#4a2040", fontsize=9)
        ax_cm.set_ylabel("Gerçek", color="#4a2040", fontsize=9)
        ax_cm.set_title("Confusion Matrix", color="#6d3b6e", fontsize=11, fontweight='bold')
        for i in range(len(labels)):
            for j in range(len(labels)):
                c = "white" if cm[i,j] > cm.max()*0.5 else "#4a2040"
                ax_cm.text(j, i, str(cm[i,j]), ha='center', va='center',
                           color=c, fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax_cm); plt.tight_layout()
        st.pyplot(fig_cm); plt.close(fig_cm)

    # MFCC-1 dağılımı
    with col_g2:
        st.markdown("#### 📊 MFCC-1 Duygu Dağılımı")
        fig_mf, ax_mf = plt.subplots(figsize=(5.5, 4.5))
        fig_mf.patch.set_facecolor("#fff0f6"); style_ax(ax_mf)
        for cls in classes_p:
            idx  = y_s == cls
            vals = X_s[idx, 0]
            if len(vals) > 0:
                ax_mf.hist(vals, bins=15, alpha=0.72,
                           label=EMO_TR.get(cls,cls).split()[0],
                           color=EMO_PALETTE.get(cls,"#ccc"),
                           edgecolor="white", linewidth=0.5)
        ax_mf.set_xlabel("MFCC-1 Değeri", color="#4a2040", fontsize=9)
        ax_mf.set_ylabel("Frekans", color="#4a2040", fontsize=9)
        ax_mf.set_title("MFCC-1 Dağılımı (Duygu Bazlı)", color="#6d3b6e",
                        fontsize=11, fontweight='bold')
        ax_mf.legend(fontsize=8, labelcolor="#4a2040",
                     facecolor="#fff0f6", edgecolor="#f8bbd0")
        plt.tight_layout()
        st.pyplot(fig_mf); plt.close(fig_mf)

    # Hata analizi
    st.divider()
    st.markdown("#### ❌ Yanlış Tahminler")
    wrong_idx = np.where(y_te_s != y_pr_s)[0]
    if len(wrong_idx) == 0:
        st.success("🎉 Hiç hata yok!")
    else:
        err_df = pd.DataFrame({
            "Gerçek": [EMO_TR.get(y_te_s[i], y_te_s[i]) for i in wrong_idx],
            "Tahmin": [EMO_TR.get(y_pr_s[i], y_pr_s[i]) for i in wrong_idx],
        })
        st.dataframe(err_df, use_container_width=True)

    # Score-Board özeti
    st.divider()
    st.markdown("#### 📋 Score-Board Özeti")
    st.markdown(f"""
    <div style='background:rgba(255,255,255,.88);border:1.5px solid #f8bbd0;
                border-radius:14px;padding:18px 24px;color:#4a2040;'>
    <b>Grup:</b> 10 &nbsp;|&nbsp; <b>Faz:</b> 1 &nbsp;|&nbsp;
    <b>Yöntem:</b> MFCC + ZCR + STE + Pitch → Random Forest<br>
    <b>Test Doğruluğu:</b>
    <span style='color:#c2185b;font-weight:800;font-size:1.1rem'>{acc_v*100:.1f}%</span>
    &nbsp;|&nbsp;
    <b>5-Fold CV:</b>
    <span style='color:#c2185b;font-weight:800;font-size:1.1rem'>
        {cv_v.mean()*100:.1f}% ± {cv_v.std()*100:.1f}%
    </span>
    </div>
    """, unsafe_allow_html=True)

    csv_data = pd.DataFrame({
        "Gerçek": y_te_s, "Tahmin": y_pr_s,
        "Doğru":  y_te_s == y_pr_s
    }).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Sonuçları CSV İndir", data=csv_data,
                       file_name="Grup10_Faz1_Sonuclar.csv", mime="text/csv")