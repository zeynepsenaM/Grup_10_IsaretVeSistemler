from flask import Flask, jsonify, render_template, send_from_directory
import numpy as np
import scipy.io.wavfile as wav
import os, webbrowser, threading
from scipy.signal import get_window, medfilt

app = Flask(__name__, template_folder='.')

# Dosya yolları ayarları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SES_KLASORU = os.path.join(BASE_DIR, "sesler")

if not os.path.exists(SES_KLASORU):
    os.makedirs(SES_KLASORU)

@app.route('/')
def index():
    return render_template('templates/index.html')

@app.route('/get_audio/<filename>')
def get_audio(filename):
    return send_from_directory(SES_KLASORU, filename)

@app.route('/list_files')
def list_files():
    # Temizlenmiş dosyaları listede gösterme
    files = [f for f in os.listdir(SES_KLASORU) if f.lower().endswith('.wav') and not f.startswith('temizlenmis_')]
    return jsonify(sorted(files))

@app.route('/analyze/<filename>')
def analyze(filename):
    try:
        file_path = os.path.join(SES_KLASORU, filename)
        fs, signal = wav.read(file_path)
        
        # 1. Orijinal Sinyal (Zaman Domeni) Verisi Hazırlama
        # Grafik performansı için seyreltme (Adım 80)
        step = max(1, len(signal) // 2000)
        raw_signal = signal[::step].astype(float).tolist()
        raw_labels = [round(i/fs, 3) for i in range(0, len(signal), step)]

        # Normalizasyon
        if signal.dtype == np.int16: 
            work_signal = signal / 32768.0 
        else:
            work_signal = signal

        # 2. Pencereleme (Hamming) - Adım 40 & 83
        f_size = int(fs * 0.02) # 20ms pencereler
        h_size = int(f_size * 0.5) # %50 örtüşme
        win = get_window('hamming', f_size)

        energies, zcrs, labels = [], [], []
        for i in range(0, len(work_signal) - f_size, h_size):
            frame = work_signal[i:i + f_size] * win # Sinyal örneği (x) ile pencereyi (w) ç
            
            # STE (Kısa Süreli Enerji) Hesaplama - Adım 46
            # Her pencereyi Hamming penceresi (win) ile çarpıp karesini alarak toplar
            #Burada frame, sinyal örneği ($x$) ile pencerenin ($w$) çarpılmış halidir.
            energies.append(float(np.sum(frame**2))) # Çarpımın karesini alıp topluyor (Sum of Squares)
            
            # ZCR (Sıfır Geçiş Oranı) Hesaplama - Adım 61
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * f_size)
            zcrs.append(float(zcr))
            
            labels.append(round(i/fs, 2))

        # 3. Dinamik Eşikleme ve VAD Kararı - Adım 87 & 89
        noise_floor = np.mean(energies[:15]) if len(energies) > 15 else 0.0001
        threshold = noise_floor * 1.3 # Hassas eşik
        
        # Medyan Filtre ile gürültü temizleme
        #Eğer enerji eşikten büyükse 1 (Konuşma), küçükse 0 (Sessizlik) etiketini verir
        vad_mask = medfilt(np.array([1.0 if e > threshold else 0.0 for e in energies]), kernel_size=3)

        colors, speech_frames_data = [], []
        speech_count = 0
        
        # 4. Renkli Maskeleme ve Konuşma Ayrıştırma
        for idx, is_speech in enumerate(vad_mask):
            start, end = idx * h_size, idx * h_size + f_size
            if is_speech == 1.0:
                speech_count += 1
                # Voiced (Yeşil) / Unvoiced (Sarı) ayrımı - Adım 62
                # ZCR yüksekse (S, Ş, F harfleri) Sarı, düşükse Yeşil
                colors.append("#3fb950" if zcrs[idx] <= 0.12 else "#f2cc60")
                speech_frames_data.append(signal[start:end])
            else:
                colors.append("#30363d") # Sessiz bölge (Gri/Siyah)

        # 5. Çıktı Üretimi: Yeni .wav kaydı 
        output_name = "temizlenmis_" + filename
        if speech_frames_data:
            cleaned_signal = np.concatenate(speech_frames_data)
            wav.write(os.path.join(SES_KLASORU, output_name), fs, cleaned_signal.astype(signal.dtype))
        else:
            output_name = None

        return jsonify({
            "raw_signal": raw_signal, "raw_labels": raw_labels,
            "orig_dur": round(len(signal)/fs, 2),
            "clean_dur": round((speech_count * h_size)/fs, 2),
            "ratio": round((1 - (speech_count * h_size)/len(signal)) * 100, 2),
            "energies": energies, "zcrs": zcrs, "labels": labels, "colors": colors,
            "cleaned_file": output_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Tarayıcıyı otomatik aç
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(port=5000)