import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import tkinter as tk

# --- MATPLOTLIB AYARLARI ---
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


class DTMFKlavye:
    def __init__(self, root):
        self.root = root
        self.root.title("DTMF Matris Klavyesi")
        self.root.configure(bg='#f8f8f8', padx=20, pady=20)

        self.row_freqs = [697, 770, 852, 941]
        self.col_freqs = [1209, 1336, 1477, 1633]

        self.color_purple = "#8E24AA"
        self.color_magenta = "#C2185B"
        self.color_pink = "#FFD1DC"

        self.create_widgets()

    def play_sound(self, freqs, title, color):
        fs = 8000
        duration = 0.5
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)

        if isinstance(freqs, (int, float)):
            signal = np.sin(2 * np.pi * freqs * t)
        else:
            signal = np.sin(2 * np.pi * freqs[0] * t) + np.sin(2 * np.pi * freqs[1] * t)

        signal = signal / np.max(np.abs(signal))

        plt.figure("DTMF Sinyal Analizi", figsize=(7, 4))
        plt.clf()
        # Çizgi kalınlığını da (linewidth) biraz artırdım ki mavi daha net görünsün
        plt.plot(t[:400], signal[:400], color=color, linewidth=1.5)
        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel("Zaman (saniye)")
        plt.ylabel("Genlik")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.pause(0.01)

        sd.play(signal, fs)

    def create_widgets(self):
        # ÜST SÜTUN BUTONLARI
        for c, f_high in enumerate(self.col_freqs):
            tk.Button(self.root,
                      text=f"{f_high} Hz",
                      font=('Times New Roman', 10, 'bold'),
                      fg=self.color_purple, bg="white",
                      activeforeground="white", activebackground=self.color_purple,
                      command=lambda fh=f_high: self.play_sound(fh, f"Tekil Yüksek Frekans: {fh}Hz", "royalblue")
                      ).grid(row=0, column=c + 1, padx=5, pady=10, sticky="nsew")

        # SOL SATIR BUTONLARI
        for r, f_low in enumerate(self.row_freqs):
            tk.Button(self.root,
                      text=f"{f_low} Hz",
                      font=('Times New Roman', 10, 'bold'),
                      fg=self.color_magenta, bg="white",
                      activeforeground="white", activebackground=self.color_magenta,
                      command=lambda fl=f_low: self.play_sound(fl, f"Tekil Düşük Frekans: {fl}Hz", "royalblue")
                      ).grid(row=r + 1, column=0, padx=10, pady=5, sticky="nsew")

        # ANA DTMF TUŞLARI
        buttons = [
            ['1', '2', '3', 'A'],
            ['4', '5', '6', 'B'],
            ['7', '8', '9', 'C'],
            ['*', '0', '#', 'D']
        ]

        for r, row in enumerate(buttons):
            for c, key in enumerate(row):
                f_low = self.row_freqs[r]
                f_high = self.col_freqs[c]

                tk.Button(self.root,
                          text=key,
                          width=8,
                          height=3,
                          bg=self.color_pink,
                          font=('Times New Roman', 12, 'bold'),
                          activebackground="#FFB7C5",
                          command=lambda k=key, fl=f_low, fh=f_high:
                          # Burada renk "#333333" yerine "royalblue" yapıldı
                          self.play_sound((fl, fh), f"Tuş {k}: {fl}Hz + {fh}Hz", "royalblue")
                          ).grid(row=r + 1, column=c + 1, padx=3, pady=3)


if __name__ == "__main__":
    plt.ion()
    root = tk.Tk()
    app = DTMFKlavye(root)
    root.mainloop()
