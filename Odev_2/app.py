import os
import io
import base64
import webbrowser
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from threading import Timer

app = Flask(__name__)

# --- TEKNİK PARAMETRELER (Raporla %100 Uyumlu) ---
FS = 16000
CHAR_DURATION = 0.04  # 40ms Sembol Süresi
GAP_DURATION = 0.02   # 20ms Boşluk (Gap)
TOTAL_STEP = CHAR_DURATION + GAP_DURATION

# Frekans Matrisi (6 Satır x 5 Sütun)
LOW_FREQS = [658, 750, 844, 936, 1030, 1126]
HIGH_FREQS = [1378, 1510, 1644, 1776, 1906]

# 29 Türk Harfi + Boşluk Karakteri
CHAR_MAP = [
    ['A', 'B', 'C', 'Ç', 'D'],
    ['E', 'F', 'G', 'Ğ', 'H'],
    ['I', 'İ', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'Ö', 'P'],
    ['R', 'S', 'Ş', 'T', 'U'],
    ['Ü', 'V', 'Y', 'Z', ' ']
]

# Klasör Yönetimi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

def manual_goertzel(samples, target_freq, fs):
    """Goertzel Algoritması: Hamming penceresi ile belirli hedef frekansın gücünü ölçer."""
    window = np.hamming(len(samples))
    samples = samples * window
    n = len(samples)
    k = int(0.5 + (n * target_freq) / fs)
    omega = (2.0 * np.pi * k) / n
    coeff = 2.0 * np.cos(omega)
    q1, q2 = 0.0, 0.0
    for x in samples:
        q0 = coeff * q1 - q2 + x
        q2, q1 = q1, q0
    return np.sqrt(q1**2 + q2**2 - q1*q2*coeff)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode():
    text = request.json.get('text', '').upper()
    if not text:
        return jsonify({"status": "error", "message": "Metin girmediniz!"})

    full_signal = []
    valid_chars = [char for row in CHAR_MAP for char in row]
    
    for char in text:
        if char not in valid_chars: continue
            
        for r, row in enumerate(CHAR_MAP):
            if char in row:
                c = row.index(char)
                t = np.linspace(0, CHAR_DURATION, int(FS * CHAR_DURATION), endpoint=False)
                # Sinyal Sentezi: s(t) = 0.5 * (sin(2*pi*f_low*t) + sin(2*pi*f_high*t))
                tone = 0.5 * (np.sin(2 * np.pi * LOW_FREQS[r] * t) + np.sin(2 * np.pi * HIGH_FREQS[c] * t))
                full_signal.extend(tone)
                full_signal.extend(np.zeros(int(FS * GAP_DURATION)))
                break

    signal_array = np.array(full_signal, dtype=np.float32)
    file_path = os.path.join(STATIC_DIR, 'output.wav')
    wav.write(file_path, FS, (signal_array * 32767).astype(np.int16))
    
    return jsonify({"status": "success", "audio_url": "/static/output.wav"})

@app.route('/decode', methods=['POST'])
def decode():
    file_path = os.path.join(STATIC_DIR, 'output.wav')
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "Önce ses üretin!"})

    samplerate, data = wav.read(file_path)
    data = data.astype(np.float32) / 32767.0
    
    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))

    decoded_text = ""
    size = int(FS * CHAR_DURATION)
    step = int(FS * TOTAL_STEP)

    for i in range(0, len(data) - size + 1, step):
        chunk = data[i: i + size]
        if np.max(np.abs(chunk)) < 0.1: continue

        low_powers = [manual_goertzel(chunk, f, FS) for f in LOW_FREQS]
        high_powers = [manual_goertzel(chunk, f, FS) for f in HIGH_FREQS]
        decoded_text += CHAR_MAP[np.argmax(low_powers)][np.argmax(high_powers)]
    
    return jsonify({"status": "success", "decoded": decoded_text})

@app.route('/analyze')
def analyze():
    file_path = os.path.join(STATIC_DIR, 'output.wav')
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "Dosya bulunamadı."})

    samplerate, data = wav.read(file_path)
    data_norm = data.astype(np.float32) / 32767.0

    # --- 3'LÜ ANALİZ PANELİ (FFT SPEKTRUM İLE) ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))
    plt.subplots_adjust(hspace=0.5)

    # 1. Grafik: Zaman Düzlemi (Waveform)
    ax1.plot(data_norm, color='#ff69b4', linewidth=0.7)
    ax1.set_title("1. Zaman Düzlemi (Waveform)", fontweight='bold')
    ax1.set_xlabel("Örnek Sayısı")
    ax1.grid(True, alpha=0.3)

    # 2. Grafik: Güç Spektrumu (FFT Magnitude Spectrum)
    n = len(data_norm)
    freqs = np.fft.rfftfreq(n, d=1/samplerate)
    magnitude = np.abs(np.fft.rfft(data_norm))
    
    ax2.plot(freqs, magnitude, color='#4B0082', linewidth=1)
    ax2.set_title("2. Güç Spektrumu (Frekans Bileşenleri)", fontweight='bold')
    ax2.set_xlim(0, 2500) # DTMF frekans aralığına zoom
    ax2.set_xlabel("Frekans (Hz)")
    ax2.set_ylabel("Genlik")
    ax2.grid(True, alpha=0.3)

    # 3. Grafik: Goertzel Güç Analizi (Hedef Frekanslardaki Enerji)
    char_samples = int(FS * CHAR_DURATION)
    first_chunk = data_norm[:char_samples]
    my_freqs = LOW_FREQS + HIGH_FREQS
    magnitudes = [manual_goertzel(first_chunk, f, FS) for f in my_freqs]
    
    colors = ['#ff99cc']*6 + ['#b284be']*5
    ax3.bar([str(f) for f in my_freqs], magnitudes, color=colors, edgecolor='black')
    ax3.set_title("3. Goertzel Filtre Analizi (Hedef Frekans İmzası)", fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)

    # Resmi Belleğe Yaz
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return jsonify({"status": "success", "plot": plot_url})

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    Timer(1.5, open_browser).start()
    app.run(debug=True, use_reloader=False)