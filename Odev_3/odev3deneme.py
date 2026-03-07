from flask import Flask, jsonify, render_template, send_from_directory
import numpy as np
import scipy.io.wavfile as wav
import os, webbrowser, threading
from scipy.signal import get_window, medfilt

app = Flask(__name__, template_folder='.')

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
    files = [f for f in os.listdir(SES_KLASORU) if f.lower().endswith('.wav') and not f.startswith('temizlenmis_')]
    return jsonify(sorted(files))

@app.route('/analyze/<filename>')
def analyze(filename):
    try:
        file_path = os.path.join(SES_KLASORU, filename)
        fs, signal = wav.read(file_path)
        
        # Sinyal Normalizasyonu (Adım 80)
        if signal.dtype == np.int16: 
            work_signal = signal / 32768.0 
        else:
            work_signal = signal

        # Pencereleme Ayarları (Adım 40 & 83)
        f_size = int(fs * 0.02)
        h_size = int(f_size * 0.5)
        win = get_window('hamming', f_size)

        energies, zcrs, labels = [], [], []
        for i in range(0, len(work_signal) - f_size, h_size):
            frame = work_signal[i:i + f_size] * win
            energies.append(float(np.sum(frame**2))) # STE (Adım 46)
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * f_size) # ZCR (Adım 61)
            zcrs.append(float(zcr))
            labels.append(round(i/fs, 2))

        # Dinamik Eşikleme (Adım 87)
        noise_floor = np.mean(energies[:15]) if len(energies) > 15 else 0.0001
        threshold = noise_floor * 1.3 

        # VAD Kararı ve Filtreleme (Adım 89)
        vad_mask = medfilt(np.array([1.0 if e > threshold else 0.0 for e in energies]), kernel_size=3)

        colors, speech_frames_data = [], []
        speech_count = 0
        
        # Konuşma Pencerelerini Ayıklama ve Renklendirme
        for idx, is_speech in enumerate(vad_mask):
            start = idx * h_size
            end = start + f_size
            
            if is_speech == 1.0:
                speech_count += 1
                # Voiced/Unvoiced Kararı (Adım 62-65)
                colors.append("#f2cc60" if zcrs[idx] > 0.12 else "#3fb950")
                speech_frames_data.append(signal[start:end])
            else:
                colors.append("#30363d")

        # Çıktı Üretimi: Temizlenmiş dosyayı kaydet
        output_name = "temizlenmis_" + filename
        if speech_frames_data:
            cleaned_signal = np.concatenate(speech_frames_data)
            output_path = os.path.join(SES_KLASORU, output_name)
            wav.write(output_path, fs, cleaned_signal.astype(signal.dtype))
        else:
            output_name = None

        return jsonify({
            "orig_dur": round(len(signal)/fs, 2),
            "clean_dur": round((speech_count * h_size)/fs, 2),
            "ratio": round((1 - (speech_count * h_size)/len(signal)) * 100, 2),
            "energies": energies, "zcrs": zcrs, "labels": labels, "colors": colors,
            "cleaned_file": output_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(port=5000)