from flask import Flask, jsonify, render_template, send_from_directory
import numpy as np
import scipy.io.wavfile as wav
import os, webbrowser, threading
from scipy.signal import medfilt

# Flask uygulamasını tanımlar ve HTML şablonlarının olduğu klasörü belirler
app = Flask(__name__, template_folder='.')

# Çalışılan dosyanın bulunduğu ana dizini alır
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ses dosyalarının saklanacağı klasör yolunu (ana_dizin/sesler) oluşturur
SES_KLASORU = os.path.join(BASE_DIR, "sesler")

# Eğer 'sesler' klasörü yoksa, çalışma hatası almamak için klasörü oluşturur
if not os.path.exists(SES_KLASORU):
    os.makedirs(SES_KLASORU)

# Tarayıcıda ana sayfa (/) açıldığında index.html dosyasını gönderir
@app.route('/')
def index():
    return render_template('templates/index.html')

# Ses dosyalarını tarayıcıda çalabilmek için dosyayı 'sesler' klasöründen sunar
@app.route('/get_audio/<filename>')
def get_audio(filename):
    return send_from_directory(SES_KLASORU, filename)

# Klasördeki .wav dosyalarını tarar ve temizlenmiş olmayanları listeler
@app.route('/list_files')
def list_files():
    # Sadece .wav uzantılı ve 'temizlenmis_' ile başlamayan dosyaları seçer
    files = [f for f in os.listdir(SES_KLASORU) if f.lower().endswith('.wav') and not f.startswith('temizlenmis_')]
    return jsonify(sorted(files)) # Dosyaları alfabetik sıralayıp JSON formatında döner

# Ana analiz fonksiyonu: Ses işleme ve boşluk silme burada yapılır
@app.route('/analyze/<filename>')
def analyze(filename):
    try:
        # Seçilen dosyanın tam yolunu oluşturur ve okur
        file_path = os.path.join(SES_KLASORU, filename)
        fs, signal = wav.read(file_path) # fs: örnekleme hızı, signal: ses verisi
        
        # 1. Sinyal Hazırlama ve Normalizasyon
        # Eğer ses 16-bit tam sayı formatındaysa (int16), -1 ile 1 arasına normalize eder
        if signal.dtype == np.int16:
            work_signal = signal.astype(np.float32) / 32768.0
        else:
            work_signal = signal.astype(np.float32)

        # 2. Parametreler (Analiz için pencere boyutları)
        win_length = int(fs * 0.02) # Analiz için 20ms'lik pencereler kullanır
        hop_length = int(fs * 0.01) # Pencereleri 10ms kaydırarak ilerler (%50 örtüşme)
        
        energies, zcrs, labels = [], [], []
        
        # Grafik için STE (Enerji) ve ZCR (Sıfır Geçiş Oranı) hesaplama döngüsü
        for i in range(0, len(work_signal) - win_length, hop_length):
            frame = work_signal[i:i + win_length] # Sinyalden bir pencere keser
            
            # Enerji: Sinyalin şiddetini RMS (kök ortalama kare) yöntemiyle hesaplar
            energies.append(float(np.sqrt(np.mean(frame**2))))
            
            # ZCR: Sinyalin sıfır noktasını geçme oranı (Sürtünmeli harfleri/gürültüyü bulur)
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * win_length)
            zcrs.append(float(zcr))
            
            # Her pencerenin zaman etiketini (saniye cinsinden) ekler
            labels.append(round(i/fs, 2))
        
        energies = np.array(energies)
        
        # 3. Dinamik Eşik ve Maskeleme (VAD - Ses Etkinliği Algılama)
        # Sesin en sessiz %10'luk kısmını 'gürültü seviyesi' olarak kabul eder
        noise_level = np.percentile(energies, 10) if len(energies) > 0 else 0.0001
        threshold = noise_level * 2.2 # Konuşmayı gürültüden ayırmak için eşik çarpanı
        
        # Enerjisi eşikten büyük olan yerleri 1 (ses), küçük olanları 0 (sessiz) yapar
        mask = (energies > threshold).astype(np.float32)
        # Medyan filtre ile maske üzerindeki küçük 'tıkırtı' gürültülerini temizler
        mask = medfilt(mask, kernel_size=5)

        # Hangover: Sesin robotik/kesik gelmemesi için konuşma bittikten sonra 150ms daha ekler
        hangover_frames = int(0.15 / (hop_length / fs))
        final_mask = np.zeros_like(mask)
        for i in range(len(mask)):
            if mask[i] == 1:
                # Konuşma tespit edilen yeri ve takip eden hangover süresini 'açık' tutar
                final_mask[i : i + hangover_frames] = 1
        
        # 4. Grafik Renkleri ve Ses Ayrıştırma
        colors = []
        keep_indices = [] # Tutulacak ses örneklerinin indis listesi
        speech_count = 0

        for i, is_speech in enumerate(final_mask[:len(energies)]):
            start = i * hop_length
            end = start + hop_length
            
            if is_speech == 1:
                speech_count += 1
                # Grafik için: ZCR düşükse yeşil (ünlüler), yüksekse sarı (sessizler) yapar
                colors.append("#3fb950" if zcrs[i] <= 0.15 else "#f2cc60")
                # Orijinal sinyaldeki bu zaman dilimine denk gelen indisleri listeye ekler
                keep_indices.extend(range(start, min(end, len(signal))))
            else:
                # Sessiz bölgeleri grafikte koyu gri gösterir
                colors.append("#30363d")

        # 5. Yeni Ses Dosyasını Oluşturma
        output_name = "temizlenmis_" + filename
        if keep_indices:
            # Sadece 'tutulacak' olarak işaretlenen indislerdeki ses verisini birleştirir
            cleaned_signal = signal[keep_indices]
            # Temizlenmiş (sessizlikleri atılmış) sesi kaydeder
            wav.write(os.path.join(SES_KLASORU, output_name), fs, cleaned_signal)
        else:
            output_name = None

        # 6. Grafik için Veri Azaltma (Seyreltme)
        # Milyonlarca veri noktasını tarayıcıyı dondurmamak için 2000 noktaya düşürür
        step = max(1, len(signal) // 2000)
        raw_signal = signal[::step].astype(float).tolist() # Seyreltilmiş sinyal genliği
        raw_labels = [round(i/fs, 3) for i in range(0, len(signal), step)] # Zaman etiketleri

        # Hesaplanan tüm analiz sonuçlarını JSON olarak tarayıcıya gönderir
        return jsonify({
            "raw_signal": raw_signal,
            "raw_labels": raw_labels,
            "orig_dur": round(len(signal)/fs, 2), # Orijinal süre
            "clean_dur": round(len(keep_indices)/fs, 2) if keep_indices else 0, # Yeni süre
            "ratio": round((1 - len(keep_indices)/len(signal)) * 100, 2) if keep_indices else 0, # Tasarruf oranı
            "energies": energies.tolist(),
            "zcrs": zcrs,
            "labels": labels,
            "colors": colors,
            "cleaned_file": output_name
        })

    except Exception as e:
        # Bir hata oluşursa terminale yazdırır ve tarayıcıya hata mesajı döner
        print(f"Hata: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Uygulama başlatıldığında çalışacak ana blok
if __name__ == '__main__':
    # Sunucu başladıktan 1.5 saniye sonra tarayıcıyı otomatik olarak açar
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    # Flask sunucusunu 5000 portunda çalıştırır
    app.run(port=5000)
if __name__ == '__main__':
    # Tarayıcıyı otomatik aç
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()

    app.run(port=5000)
