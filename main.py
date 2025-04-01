import numpy as np
import scipy.signal as sign
import cohen as hallelujah
import matplotlib.pyplot as plt
import scipy as sc

N, f_s = 512, 500
t = np.arange(N) / f_s

def WV_and_specgram(signal, num):  # 3.3
    fig, ax = plt.subplots(nrows=2, ncols=1, layout='constrained', num=num)
    hilbert = sign.hilbert(signal)
    WignerVille, fWV, tWV = hallelujah.cohen(hilbert, fs=f_s)
    ax[1].pcolormesh(tWV, fWV, np.abs(WignerVille))
    ax[0].specgram(signal, NFFT=64, Fs=f_s, noverlap=32)
    fig.savefig("img/"+num+".png")

# ______________ defining chirps ________________________________
chirp = sign.chirp(t, f0=20, f1=100, t1=N / f_s, ) # 3.1
chirp200 = sign.chirp(t, f0=20, f1=200, t1=N / f_s, ) # 3.4a
chirp300 = sign.chirp(t, f0=20, f1=300, t1=N / f_s, ) # 3.4a
chirp400 = sign.chirp(t, f0=20, f1=400, t1=N / f_s, ) # 3.4a
chirp500 = sign.chirp(t, f0=20, f1=500, t1=N / f_s, ) # 3.4a
chirp_sum_par = sign.chirp(t, f0=20, f1=100, t1=N / f_s, ) + sign.chirp(t, f0=40, f1=120, t1=N / f_s, ) # 3.4b
chirp_sum_nonpar = sign.chirp(t, f0=20, f1=100, t1=N / f_s, ) + sign.chirp(t, f0=40, f1=200, t1=N / f_s, ) # 3.4c

lenghts = [16, 32, 64, 128, 256]

# ______________ plotting spectrograms ___________________________
fig, ax = plt.subplots(nrows=3, ncols=2, layout='constrained', num="Chirps")
for y in range(3): # 3.2
    for x in range(2):
        if x + y == 0:
            ax[0][0].plot(t, chirp)
        else:
            f, t, Sxx = sc.signal.spectrogram(chirp, f_s, window=sign.get_window('boxcar', lenghts[x + 2 * y - 1]))
            ax[y][x].set_title(str(lenghts[y * 2 + x - 1]))
            ax[y][x].pcolormesh(t, f, Sxx, shading='gouraud')
            ax[y][x].set_ylabel('Frequency [Hz]')
            ax[y][x].set_xlabel('Time [sec]')
fig.savefig("img/chirp.png")

# ______________ Wigner-Ville and specgram for nfft=64 ________________________________
WV_and_specgram(chirp, "WV - Base") # 3.3
WV_and_specgram(chirp200, "WV - 200Hz") # 3.4a
WV_and_specgram(chirp300, "WV - 300Hz") # 3.4a
WV_and_specgram(chirp400, "WV - 400Hz") # 3.4a
WV_and_specgram(chirp500, "WV - 500Hz") # 3.4a

WV_and_specgram(chirp_sum_par, "WV - Paralel sum") # 3.4b
WV_and_specgram(chirp_sum_nonpar, "WV - non-paralel sum") # 3.4c

plt.show()
