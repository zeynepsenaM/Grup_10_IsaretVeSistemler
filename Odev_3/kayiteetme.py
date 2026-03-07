import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

# Ayarlar
fs = 44100  # Örnekleme hızı (Ödev standartı)
kanallar = 1 # Mono kayıt

print("--- KAYIT BAŞLIYOR ---")
print("Konuşman bittiğinde durdurmak için 'Enter' tuşuna bas...")

# Boş bir liste oluşturup veriyi burada toplayacağız
kayit_verisi = []

def callback(indata, frames, time, status):
    if status:
        print(status)
    kayit_verisi.append(indata.copy())

# Kaydı başlatıyoruz
with sd.InputStream(samplerate=fs, channels=kanallar, callback=callback):
    input() # Kullanıcı Enter'a basana kadar burada bekler

# Kaydedilen parçaları birleştiriyoruz
final_kayit = np.concatenate(kayit_verisi, axis=0)

print(f"Kayıt tamamlandı. Toplam süre: {len(final_kayit)/fs:.2f} saniye.")
write("seskaydi.wav", fs, final_kayit)
print("'seskaydi.wav' dosyası başarıyla oluşturuldu.")