"""
BIL216 - İşaretler ve Sistemler
2025-2026 Bahar Dönemi | Grup 10
Dönemiçi Proje: Ses İşareti Analizi ve Cinsiyet Sınıflandırma
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob

# ─────────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ses Cinsiyet Sınıflandırıcı | Grup 10",
    page_icon="🎙️",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS – Pastel Pembe Tasarım
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
    }

    /* Arka plan: açık krem-pembe gradient */
    .stApp {
        background: linear-gradient(160deg, #fff0f6, #fce4ec, #f3e5f5, #ede7f6);
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.75) !important;
        backdrop-filter: blur(12px);
        border-right: 1.5px solid #f8bbd0;
    }
    [data-testid="stSidebar"] * {
        color: #6d3b6e !important;
    }

    /* Genel yazı rengi */
    .stApp, .stApp p, .stApp li, .stApp label,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #4a2040 !important;
    }

    /* Metrik kartları */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.85);
        border: 1.5px solid #f8bbd0;
        border-radius: 18px;
        padding: 16px 20px;
        box-shadow: 0 4px 16px rgba(233,30,99,0.08);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        color: #c2185b !important;
    }
    [data-testid="stMetricLabel"] {
        color: #ad6f8a !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }

    /* Sekmeler */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.7);
        border-radius: 14px;
        padding: 5px;
        gap: 4px;
        border: 1.5px solid #f8bbd0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #ad6f8a;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #f48fb1, #ce93d8) !important;
        color: white !important;
        box-shadow: 0 3px 12px rgba(244,143,177,0.4);
    }

    /* Butonlar */
    .stButton > button {
        background: linear-gradient(90deg, #f48fb1, #ce93d8);
        color: white !important;
        border: none;
        border-radius: 14px;
        font-weight: 700;
        font-size: 1rem;
        padding: 10px 28px;
        box-shadow: 0 4px 18px rgba(244,143,177,0.4);
        transition: all 0.25s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(244,143,177,0.55);
    }

    /* Input alanları */
    .stTextInput > div > div > input {
        background: white !important;
        border: 1.5px solid #f8bbd0 !important;
        border-radius: 10px !important;
        color: #4a2040 !important;
        font-weight: 600 !important;
    }
    .stSelectbox > div > div {
        background: white !important;
        border: 1.5px solid #f8bbd0 !important;
        border-radius: 10px !important;
        color: #4a2040 !important;
    }

    /* Slider */
    [data-testid="stSlider"] > div > div > div {
        background: linear-gradient(90deg, #f48fb1, #ce93d8) !important;
    }

    /* Divider */
    hr { border-color: #f8bbd0 !important; }

    /* Info / success / warning kutuları */
    .stAlert {
        border-radius: 14px !important;
        border: none !important;
    }

    /* Tablo */
    .stDataFrame { border-radius: 14px; overflow: hidden; }

    /* Başlık */
    .hero-title {
        background: linear-gradient(90deg, #e91e63, #9c27b0, #e91e63);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 50px !important;
        font-weight: 800 !important;
        margin-bottom: 5px !important;
        line-height: 1.2 !important;
        display: block !important;
        text-align: center !important;
    }
    .hero-sub {
        color: #ad6f8a;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 2px;
    }

    /* Cinsiyet badge */
    .badge-E {
        background: linear-gradient(90deg, #90caf9, #42a5f5);
        color: white; padding: 7px 22px; border-radius: 22px;
        font-weight: 800; font-size: 1.05rem;
        box-shadow: 0 3px 10px rgba(66,165,245,0.3);
    }
    .badge-K {
        background: linear-gradient(90deg, #f48fb1, #e91e63);
        color: white; padding: 7px 22px; border-radius: 22px;
        font-weight: 800; font-size: 1.05rem;
        box-shadow: 0 3px 10px rgba(233,30,99,0.3);
    }
    .badge-C {
        background: linear-gradient(90deg, #a5d6a7, #66bb6a);
        color: white; padding: 7px 22px; border-radius: 22px;
        font-weight: 800; font-size: 1.05rem;
        box-shadow: 0 3px 10px rgba(102,187,106,0.3);
    }
    .badge-? {
        background: #eeeeee; color: #777;
        padding: 7px 22px; border-radius: 22px;
        font-weight: 700; font-size: 1.05rem;
    }

    /* Kart kutusu */
    .info-box {
        background: rgba(255,255,255,0.85);
        border: 1.5px solid #f8bbd0;
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 12px rgba(233,30,99,0.06);
        color: #4a2040;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BAŞLIK
# ─────────────────────────────────────────────
st.markdown('<p class="hero-title">🎙️ Ses İşareti Analizi ve Cinsiyet Sınıflandırma Sistemi </p>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# SABİTLER
# ─────────────────────────────────────────────
FRAME_MS               = 30
MIN_F0                 = 60
MAX_F0                 = 500
THRESHOLD_MALE_FEMALE  = 160
THRESHOLD_FEMALE_CHILD = 260

# ─────────────────────────────────────────────
# FONKSİYONLAR
# ─────────────────────────────────────────────

def normalize_audio(y):
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y

def compute_ste(y, frame_length, hop_length):
    return np.array([
        np.sum(y[i:i+frame_length]**2)
        for i in range(0, len(y)-frame_length, hop_length)
    ])

def compute_zcr(y, frame_length, hop_length):
    return librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length
    )[0]

def detect_voiced_frames(ste, zcr, ste_ratio=0.02, zcr_thresh=0.15):
    min_len    = min(len(ste), len(zcr))
    ste, zcr   = ste[:min_len], zcr[:min_len]
    ste_thresh = ste_ratio * np.max(ste) if np.max(ste) > 0 else 0
    return (ste > ste_thresh) & (zcr < zcr_thresh)

def autocorrelation_f0(frame, sr):
    n        = len(frame)
    r        = np.correlate(frame, frame, mode='full')[n-1:]
    r        = r / (r[0] + 1e-10)
    lag_min  = int(sr / MAX_F0)
    lag_max  = int(sr / MIN_F0)
    if lag_max >= len(r) or lag_min >= lag_max:
        return None
    search   = r[lag_min:lag_max]
    peak_idx = np.argmax(search)
    if search[peak_idx] < 0.3:
        return None
    return sr / (lag_min + peak_idx)

def extract_features(y, sr):
    y            = normalize_audio(y)
    frame_length = int(sr * FRAME_MS / 1000)
    hop_length   = frame_length // 2
    ste          = compute_ste(y, frame_length, hop_length)
    zcr          = compute_zcr(y, frame_length, hop_length)
    voiced       = detect_voiced_frames(ste, zcr)
    f0_list      = []
    frames       = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    for i, frame in enumerate(frames.T):
        if i < len(voiced) and voiced[i]:
            f0 = autocorrelation_f0(frame, sr)
            if f0:
                f0_list.append(f0)
    mean_f0 = np.mean(f0_list) if f0_list else np.nan
    std_f0  = np.std(f0_list)  if f0_list else np.nan
    return {
        "mean_f0": mean_f0, "std_f0": std_f0,
        "mean_zcr": float(np.mean(zcr)),
        "mean_ste": float(np.mean(ste)),
        "voiced_ratio": float(np.sum(voiced) / max(len(voiced), 1)),
        "f0_values": f0_list, "ste": ste, "zcr": zcr
    }

def classify(mean_f0, thresh_mf, thresh_fc):
    if np.isnan(mean_f0):      return "?"
    if mean_f0 < thresh_mf:    return "E"
    if mean_f0 <= thresh_fc:   return "K"
    return "C"

def normalize_gender_label(raw):
    """Türkçe (E/K/C) ve İngilizce (M/F/C) etiketleri ortak formata çevirir."""
    mapping = {
        "E": "E", "K": "K", "C": "C",   # Türkçe format
        "M": "E", "F": "K",              # İngilizce format (M→Erkek, F→Kadın)
    }
    return mapping.get(str(raw).upper().strip(), "?")

def label_tr(code):
    return {"E": "Erkek 👨", "K": "Kadın 👩", "C": "Çocuk 🧒", "?": "Bilinmiyor"}.get(code, code)

def badge(code):
    cls = {"E": "badge-E", "K": "badge-K", "C": "badge-C"}.get(code, "badge-?")
    return f'<span class="{cls}">{label_tr(code)}</span>'

def style_ax(ax):
    """Açık arka planlı, koyu yazılı grafik stili"""
    ax.set_facecolor("#fff5f8")
    ax.tick_params(colors='#4a2040', labelsize=8)
    ax.xaxis.label.set_color('#4a2040')
    ax.yaxis.label.set_color('#4a2040')
    ax.title.set_color('#6d3b6e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#f8bbd0')

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    dataset_path = st.text_input("📁 Dataset klasör yolu", value="c:\Python\IsaretVeSistemler\Odev_1\DONEM_ODEVI\Dataset")
    excel_path   = st.text_input("📋 Excel (metadata) yolu", value="c:\Python\IsaretVeSistemler\Odev_1\DONEM_ODEVI\metadata.xlsx")

    st.divider()
    st.markdown("### 🎚️ Sınıflandırıcı Eşikleri")
    thresh_mf = st.slider("Erkek / Kadın eşiği (Hz)", 100, 220, THRESHOLD_MALE_FEMALE, 5)
    thresh_fc = st.slider("Kadın / Çocuk eşiği (Hz)", 200, 400, THRESHOLD_FEMALE_CHILD, 5)
    st.markdown(f"""
    <div style='background:rgba(244,143,177,0.12);border:1.5px solid #f8bbd0;
                border-radius:12px;padding:12px;font-size:0.85rem;
                color:#6d3b6e;text-align:center;font-weight:600;'>
    🔵 Erkek &lt; {thresh_mf} Hz<br>
    🩷 {thresh_mf}–{thresh_fc} Hz = Kadın<br>
    🟢 Çocuk &gt; {thresh_fc} Hz
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🎵 Ses Dosyaları")
    wav_files_all = glob.glob(os.path.join(dataset_path, "**", "*.wav"), recursive=True)
    wav_names     = sorted([os.path.basename(f) for f in wav_files_all])
    wav_map       = {os.path.basename(f): f for f in wav_files_all}
    if wav_names:
        st.success(f"✅ {len(wav_names)} dosya bulundu")
    else:
        st.warning("⚠️ Dataset klasöründe .wav bulunamadı")

# ─────────────────────────────────────────────
# SEKMELER
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Tek Dosya Analizi",
    "📊 Veri Seti Analizi",
    "📈 İstatistikler & Başarı"
])

# ══════════════════════════════════════════════
# TAB 1 – TEK DOSYA ANALİZİ
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### 🔍 Tek Ses Dosyası Analizi")

    source = st.radio("Kaynak seçin", ["📂 Dataset'ten seç", "⬆️ Dosya yükle"], horizontal=True)

    y_input, sr_input, fname_input = None, None, ""

    if source == "📂 Dataset'ten seç":
        if wav_names:
            selected = st.selectbox("Ses dosyası seçin", wav_names,
                                    format_func=lambda x: f"🎵 {x}")
            if st.button("▶️ Analiz Et"):
                y_input, sr_input = librosa.load(wav_map[selected], sr=None, mono=True)
                fname_input       = selected
                st.session_state["single_y"]  = y_input
                st.session_state["single_sr"] = sr_input
                st.session_state["single_fn"] = fname_input
        else:
            st.warning("Dataset klasöründe dosya bulunamadı. Sol panelden yolu kontrol edin.")
    else:
        uploaded = st.file_uploader("Bir .wav dosyası yükleyin", type=["wav"])
        if uploaded:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            y_input, sr_input = librosa.load(tmp_path, sr=None, mono=True)
            fname_input       = uploaded.name
            os.unlink(tmp_path)
            st.session_state["single_y"]  = y_input
            st.session_state["single_sr"] = sr_input
            st.session_state["single_fn"] = fname_input

    if "single_y" in st.session_state:
        y_input     = st.session_state["single_y"]
        sr_input    = st.session_state["single_sr"]
        fname_input = st.session_state["single_fn"]

    if y_input is not None:
        with st.spinner("🔬 Analiz yapılıyor..."):
            feats = extract_features(y_input, sr_input)
            pred  = classify(feats["mean_f0"], thresh_mf, thresh_fc)

        parts      = fname_input.replace(".wav", "").split("_")
        true_label = normalize_gender_label(parts[2]) if len(parts) >= 3 else "?"

        st.divider()

        # Badge satırı
        col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
        with col_b1:
            st.markdown("**🎯 Tahmin:**")
            st.markdown(badge(pred), unsafe_allow_html=True)
        with col_b2:
            st.markdown("**✅ Gerçek:**")
            st.markdown(badge(true_label), unsafe_allow_html=True)
        with col_b3:
            match = "✅ Doğru!" if pred == true_label else "❌ Yanlış"
            color = "#2e7d32" if pred == true_label else "#c62828"
            st.markdown(
                f"**Sonuç:** <span style='color:{color};font-weight:800;font-size:1.2rem'>{match}</span>",
                unsafe_allow_html=True
            )

        st.divider()

        # Metrikler
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎵 Ortalama F0",  f"{feats['mean_f0']:.1f} Hz" if not np.isnan(feats['mean_f0']) else "N/A")
        c2.metric("📊 F0 Std Sapma", f"{feats['std_f0']:.1f} Hz"  if not np.isnan(feats['std_f0'])  else "N/A")
        c3.metric("〰️ ZCR",           f"{feats['mean_zcr']:.4f}")
        c4.metric("🔊 Sesli Oran",    f"{feats['voiced_ratio']*100:.1f}%")

        st.divider()

        # ─── 6'lı grafik paneli ───
        COLORS = {
            "wave":  "#e91e63",
            "ste":   "#9c27b0",
            "zcr":   "#f06292",
            "acorr": "#7b1fa2",
            "fft":   "#c2185b",
            "f0":    "#ab47bc",
        }
        BG_FIG = "#fff0f6"
        BG_AX  = "#fff5f8"

        fig = plt.figure(figsize=(16, 9))
        fig.patch.set_facecolor(BG_FIG)
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

        t = np.linspace(0, len(y_input)/sr_input, len(y_input))

        # 1. Dalga Formu
        ax1 = fig.add_subplot(gs[0, 0])
        style_ax(ax1)
        ax1.plot(t, y_input, color=COLORS["wave"], linewidth=0.5)
        ax1.fill_between(t, y_input, alpha=0.18, color=COLORS["wave"])
        ax1.set_title("🌊 Dalga Formu", fontsize=10, fontweight='bold')
        ax1.set_xlabel("Zaman (s)", fontsize=8)
        ax1.set_ylabel("Genlik", fontsize=8)

        # 2. STE
        ax2 = fig.add_subplot(gs[0, 1])
        style_ax(ax2)
        ste_t = np.linspace(0, len(y_input)/sr_input, len(feats["ste"]))
        ax2.plot(ste_t, feats["ste"], color=COLORS["ste"], linewidth=1.3)
        ax2.fill_between(ste_t, feats["ste"], alpha=0.18, color=COLORS["ste"])
        ax2.set_title("⚡ Kısa Süreli Enerji (STE)", fontsize=10, fontweight='bold')
        ax2.set_xlabel("Zaman (s)", fontsize=8)

        # 3. ZCR
        ax3 = fig.add_subplot(gs[0, 2])
        style_ax(ax3)
        zcr_t = np.linspace(0, len(y_input)/sr_input, len(feats["zcr"]))
        ax3.plot(zcr_t, feats["zcr"], color=COLORS["zcr"], linewidth=1.3)
        ax3.fill_between(zcr_t, feats["zcr"], alpha=0.18, color=COLORS["zcr"])
        ax3.set_title("〰️ Sıfır Geçiş Oranı (ZCR)", fontsize=10, fontweight='bold')
        ax3.set_xlabel("Zaman (s)", fontsize=8)

        # 4. Otokorelasyon
        ax4 = fig.add_subplot(gs[1, 0])
        style_ax(ax4)
        fl = int(sr_input * 0.03)
        if len(y_input) >= fl:
            frame_ac = y_input[:fl]
            n        = len(frame_ac)
            r        = np.correlate(frame_ac, frame_ac, mode='full')[n-1:]
            r        = r / (r[0] + 1e-10)
            lag_ms   = np.arange(len(r)) / sr_input * 1000
            show     = int(sr_input * 0.02)
            ax4.plot(lag_ms[:show], r[:show], color=COLORS["acorr"], linewidth=1.5)
            if not np.isnan(feats["mean_f0"]) and feats["mean_f0"] > 0:
                peak_ms = 1000 / feats["mean_f0"]
                ax4.axvline(peak_ms, color='#e91e63', linewidth=2, linestyle='--',
                            label=f"F0 ≈ {feats['mean_f0']:.0f} Hz")
                ax4.legend(fontsize=8, labelcolor='#4a2040',
                           facecolor='#fff0f6', edgecolor='#f8bbd0')
        ax4.set_title("🔄 Otokorelasyon R(τ)", fontsize=10, fontweight='bold')
        ax4.set_xlabel("Gecikme τ (ms)", fontsize=8)

        # 5. FFT
        ax5 = fig.add_subplot(gs[1, 1])
        style_ax(ax5)
        N       = min(len(y_input), 8192)
        fft_mag = np.abs(np.fft.rfft(y_input[:N], n=N))
        freqs   = np.fft.rfftfreq(N, d=1/sr_input)
        mask    = freqs <= 1200
        ax5.plot(freqs[mask], fft_mag[mask], color=COLORS["fft"], linewidth=1.1)
        ax5.fill_between(freqs[mask], fft_mag[mask], alpha=0.18, color=COLORS["fft"])
        if not np.isnan(feats["mean_f0"]):
            ax5.axvline(feats["mean_f0"], color='#9c27b0', linewidth=2, linestyle='--',
                        label=f"F0 ≈ {feats['mean_f0']:.0f} Hz")
            ax5.legend(fontsize=8, labelcolor='#4a2040',
                       facecolor='#fff0f6', edgecolor='#f8bbd0')
        ax5.set_title("📡 FFT Büyüklük Spektrumu", fontsize=10, fontweight='bold')
        ax5.set_xlabel("Frekans (Hz)", fontsize=8)

        # 6. F0 Zaman Serisi
        ax6 = fig.add_subplot(gs[1, 2])
        style_ax(ax6)
        if feats["f0_values"]:
            ax6.plot(feats["f0_values"], color=COLORS["f0"], linewidth=1.5,
                     marker='o', markersize=3, markerfacecolor='#e91e63', markeredgewidth=0)
            ax6.axhline(thresh_mf, color='#42a5f5', linestyle='--', linewidth=1.4,
                        label=f'E/K ({thresh_mf} Hz)')
            ax6.axhline(thresh_fc, color='#66bb6a', linestyle='--', linewidth=1.4,
                        label=f'K/Ç ({thresh_fc} Hz)')
            ax6.axhspan(0,         thresh_mf, alpha=0.07, color='#42a5f5')
            ax6.axhspan(thresh_mf, thresh_fc, alpha=0.07, color='#f48fb1')
            ax6.axhspan(thresh_fc, 500,       alpha=0.07, color='#66bb6a')
            ax6.legend(fontsize=7, labelcolor='#4a2040',
                       facecolor='#fff0f6', edgecolor='#f8bbd0')
        ax6.set_title("🎼 F0 Zaman Serisi", fontsize=10, fontweight='bold')
        ax6.set_xlabel("Çerçeve", fontsize=8)
        ax6.set_ylabel("F0 (Hz)", fontsize=8)

        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════
# TAB 2 – VERİ SETİ ANALİZİ
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Tüm Veri Seti Üzerinde Analiz")

    col_i1, col_i2, col_i3 = st.columns(3)
    col_i1.metric("📁 Dataset Yolu", dataset_path)
    col_i2.metric("📋 Excel Yolu",   excel_path)
    col_i3.metric("🎵 Bulunan .wav", len(wav_names))

    st.divider()

    if st.button("🚀 Veri Seti Analizini Başlat", type="primary"):
        if not os.path.exists(excel_path):
            st.error(f"❌ Excel dosyası bulunamadı: `{excel_path}`")
            st.stop()
        if not wav_names:
            st.error(f"❌ Dataset klasöründe .wav bulunamadı: `{dataset_path}`")
            st.stop()

        df = pd.read_excel(excel_path, sheet_name=0)
        df = df.dropna(subset=["Dosya_Adi"])
        df["Dosya_Adi"] = df["Dosya_Adi"].astype(str).str.strip()
        df["Cinsiyet"] = df["Cinsiyet"].astype(str).str.upper().str.strip()
        st.success(f"✅ Excel yüklendi: **{len(df)}** kayıt | **{len(wav_names)}** ses dosyası")

        results       = []
        progress_bar  = st.progress(0)
        progress_text = st.empty()
        total         = len(df)

        for idx, row in df.iterrows():
            fname  = str(row["Dosya_Adi"]).strip()
            true_c = normalize_gender_label(row["Cinsiyet"])
            progress_bar.progress((idx+1)/total)
            progress_text.markdown(f"⏳ **{idx+1}/{total}** — `{fname}`")

            if fname not in wav_map:
                results.append({"Dosya_Adi": fname, "Gercek": true_c, "Tahmin": "YOK",
                                 "Dogru": False, "mean_f0": np.nan, "std_f0": np.nan, "mean_zcr": np.nan})
                continue
            try:
                y, sr = librosa.load(wav_map[fname], sr=None, mono=True)
                feats = extract_features(y, sr)
                pred  = classify(feats["mean_f0"], thresh_mf, thresh_fc)
                results.append({"Dosya_Adi": fname, "Gercek": true_c, "Tahmin": pred,
                                 "Dogru": pred == true_c, "mean_f0": feats["mean_f0"],
                                 "std_f0": feats["std_f0"], "mean_zcr": feats["mean_zcr"]})
            except Exception:
                results.append({"Dosya_Adi": fname, "Gercek": true_c, "Tahmin": "HATA",
                                 "Dogru": False, "mean_f0": np.nan, "std_f0": np.nan, "mean_zcr": np.nan})

        progress_bar.empty()
        progress_text.empty()

        results_df = pd.DataFrame(results)
        st.session_state["results_df"] = results_df

        valid = results_df[results_df["Tahmin"].isin(["E", "K", "C"])]
        acc   = valid["Dogru"].mean() * 100

        st.balloons()
        st.success(f"✅ Analiz tamamlandı! Genel Doğruluk: **{acc:.1f}%**")
        st.dataframe(
            results_df[["Dosya_Adi", "Gercek", "Tahmin", "Dogru", "mean_f0"]]
            .rename(columns={"mean_f0": "F0 (Hz)", "Gercek": "Gerçek"}),
            use_container_width=True
        )


# ══════════════════════════════════════════════
# TAB 3 – İSTATİSTİKLER & BAŞARI
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 İstatistiksel Bulgular ve Başarı Analizi")

    if "results_df" not in st.session_state:
        st.info("ℹ️ Önce **Veri Seti Analizi** sekmesinden analizi çalıştırın.")
        st.stop()

    rdf         = st.session_state["results_df"]
    valid       = rdf[rdf["Tahmin"].isin(["E", "K", "C"])].copy()
    overall_acc = valid["Dogru"].mean() * 100

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("🎯 Genel Doğruluk", f"{overall_acc:.1f}%")
    col_b.metric("📁 Toplam Dosya",   len(rdf))
    col_c.metric("✅ Doğru Tahmin",   int(valid["Dogru"].sum()))
    col_d.metric("❌ Yanlış Tahmin",  int((~valid["Dogru"]).sum()))

    st.divider()

    st.markdown("#### 📋 Sınıf Bazlı İstatistikler")
    rows_table = []
    for cls in ["E", "K", "C"]:
        sub   = valid[valid["Gercek"] == cls]
        acc_c = sub["Dogru"].mean() * 100 if len(sub) > 0 else 0
        rows_table.append({
            "Sınıf":            label_tr(cls),
            "Örnek Sayısı":     len(sub),
            "Ortalama F0 (Hz)": f"{sub['mean_f0'].mean():.1f}" if len(sub) > 0 else "-",
            "Standart Sapma":   f"{sub['mean_f0'].std():.1f}"  if len(sub) > 0 else "-",
            "Başarı (%)":       f"{acc_c:.1f}%"
        })
    st.table(pd.DataFrame(rows_table))

    st.divider()

    col_g1, col_g2 = st.columns(2)

    # Confusion Matrix
    with col_g1:
        st.markdown("#### 🔲 Karışıklık Matrisi")
        labels      = ["E", "K", "C"]
        label_names = ["Erkek", "Kadın", "Çocuk"]
        cm = np.zeros((3, 3), dtype=int)
        for i, tc in enumerate(labels):
            for j, pc in enumerate(labels):
                cm[i, j] = ((valid["Gercek"] == tc) & (valid["Tahmin"] == pc)).sum()

        fig3, ax3 = plt.subplots(figsize=(5, 4))
        fig3.patch.set_facecolor("#fff0f6")
        style_ax(ax3)
        im = ax3.imshow(cm, cmap='RdPu')
        ax3.set_xticks([0, 1, 2]); ax3.set_yticks([0, 1, 2])
        ax3.set_xticklabels(label_names, color='#4a2040', fontsize=10, fontweight='bold')
        ax3.set_yticklabels(label_names, color='#4a2040', fontsize=10, fontweight='bold')
        ax3.set_xlabel("Tahmin Edilen", color='#4a2040', fontsize=10)
        ax3.set_ylabel("Gerçek",        color='#4a2040', fontsize=10)
        ax3.set_title("Confusion Matrix", color='#6d3b6e', fontsize=11, fontweight='bold')
        for i in range(3):
            for j in range(3):
                color = 'white' if cm[i, j] > cm.max() * 0.5 else '#4a2040'
                ax3.text(j, i, str(cm[i, j]), ha='center', va='center',
                         color=color, fontsize=16, fontweight='bold')
        plt.colorbar(im, ax=ax3)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # F0 Dağılımı
    with col_g2:
        st.markdown("#### 📊 F0 Dağılımı")
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        fig4.patch.set_facecolor("#fff0f6")
        style_ax(ax4)
        colors_cls = {"E": "#90caf9", "K": "#f48fb1", "C": "#a5d6a7"}
        for cls in ["E", "K", "C"]:
            sub = valid[valid["Gercek"] == cls]["mean_f0"].dropna()
            if len(sub) > 0:
                ax4.hist(sub, bins=20, alpha=0.78,
                         label=label_tr(cls).split()[0],
                         color=colors_cls[cls], edgecolor='white', linewidth=0.5)
        ax4.axvline(thresh_mf, color='#42a5f5', linestyle='--', linewidth=1.8,
                    label=f'E/K ({thresh_mf} Hz)')
        ax4.axvline(thresh_fc, color='#66bb6a', linestyle='--', linewidth=1.8,
                    label=f'K/Ç ({thresh_fc} Hz)')
        ax4.set_xlabel("Ortalama F0 (Hz)", color='#4a2040', fontsize=9)
        ax4.set_ylabel("Frekans",          color='#4a2040', fontsize=9)
        ax4.set_title("Sınıf Bazlı F0 Dağılımı", color='#6d3b6e', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8, labelcolor='#4a2040',
                   facecolor='#fff0f6', edgecolor='#f8bbd0')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    # Hata Analizi
    st.divider()
    st.markdown("#### ❌ Hata Analizi")
    errors = valid[~valid["Dogru"]][["Dosya_Adi", "Gercek", "Tahmin", "mean_f0"]].copy()
    if len(errors) == 0:
        st.success("🎉 Hiç hata yok!")
    else:
        errors = errors.rename(columns={"mean_f0": "F0 (Hz)", "Gercek": "Gerçek"})
        errors["Gerçek"] = errors["Gerçek"].map(lambda x: label_tr(x).split()[0])
        errors["Tahmin"] = errors["Tahmin"].map(lambda x: label_tr(x).split()[0])
        st.dataframe(errors, use_container_width=True)

    st.divider()
    csv = valid.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Sonuçları CSV olarak indir",
        data=csv,
        file_name="Grup10_Sonuclar.csv",
        mime="text/csv"
    )