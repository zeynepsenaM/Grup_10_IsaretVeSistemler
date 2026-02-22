import numpy as np
import matplotlib.pyplot as plt

# 1. Parametrelerin Belirlenmesi
f0 = 59
f1 = f0
f2 = f0 / 2
f3 = 10 * f0

# Örnekleme Frekansı (Nyquist: fs > 2 * f3)
# f3 = 590 Hz olduğu için fs en az 1180 Hz olmalı. 
# Pürüzsüz bir görüntü için 20 katını (11800 Hz) seçiyoruz.
fs = 20 * f3 

# 2. Zaman Ekseni (Dinamik Ayarlama)
# En düşük frekans f2 (29.5 Hz) olduğu için, 3 periyot görmek adına:
# Periyot T = 1/f -> 3 periyot = 3/f
T_max = 3 / f2 
t = np.linspace(0, T_max, int(fs * T_max))

# Sinyal Üretimi
y1 = np.sin(2 * np.pi * f1 * t)
y2 = np.sin(2 * np.pi * f2 * t)
y3 = np.sin(2 * np.pi * f3 * t)
y_sum = y1 + y2 + y3

# 3. Görselleştirme
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Sinyal 1
axs[0].plot(t, y1, color='blue')
axs[0].set_title(f'Sinyal 1 (f1 = {f1} Hz)')
axs[0].grid(True)

# Sinyal 2
axs[1].plot(t, y2, color='green')
axs[1].set_title(f'Sinyal 2 (f2 = {f2} Hz)')
axs[1].grid(True)

# Sinyal 3
axs[2].plot(t, y3, color='red')
axs[2].set_title(f'Sinyal 3 (f3 = {f3} Hz)')
axs[2].grid(True)

# Toplam Sinyal
axs[3].plot(t, y_sum, color='black')
axs[3].set_title('Üç Sinyalin Toplamı (y1 + y2 + y3)')
axs[3].set_xlabel('Zaman (saniye)')
axs[3].grid(True)

plt.tight_layout()
plt.show()