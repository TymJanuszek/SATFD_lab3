import numpy as np
import scipy.signal as sign
import cohen as hallelujah
import matplotlib.pyplot as plt
import scipy as sc

N, f_s = 512, 500
t = np.arange(N) / f_s


# fig, ax = plt.subplots(nrows=2, ncols=1, layout='constrained', num="Chirps")

def WV_and_specgram(signal, num):
    fig, ax = plt.subplots(nrows=2, ncols=1, layout='constrained', num=num)
    hilbert = sign.hilbert(signal)
    WignerVille, fWV, tWV = hallelujah.cohen(hilbert, fs=f_s)
    ax[1].pcolormesh(tWV, fWV, np.abs(WignerVille))
    ax[0].specgram(signal, NFFT=64, Fs=f_s, noverlap=32)
    plt.savefig(fig)

chirp = sign.chirp(t, f0=20, f1=100, t1=N / f_s, )
chirp200 = sign.chirp(t, f0=20, f1=200, t1=N / f_s, )
chirp300 = sign.chirp(t, f0=20, f1=300, t1=N / f_s, )
chirp400 = sign.chirp(t, f0=20, f1=400, t1=N / f_s, )
chirp500 = sign.chirp(t, f0=20, f1=500, t1=N / f_s, )
chirp_sum_par = sign.chirp(t, f0=20, f1=100, t1=N / f_s, ) + sign.chirp(t, f0=40, f1=120, t1=N / f_s, )
chirp_sum_nonpar = sign.chirp(t, f0=20, f1=100, t1=N / f_s, ) + sign.chirp(t, f0=40, f1=200, t1=N / f_s, )

lenghts = [16, 32, 64, 128, 256]

fig, ax = plt.subplots(nrows=3, ncols=2, layout='constrained', num="Chirps")
for y in range(3):
    for x in range(2):
        if x + y == 0:
            ax[0][0].plot(t, chirp)
        else:
            f, t, Sxx = sc.signal.spectrogram(chirp, f_s, window=sign.get_window('boxcar', lenghts[x + 2 * y - 1]))
            ax[y][x].set_title(str(lenghts[y * 2 + x - 1]))
            ax[y][x].pcolormesh(t, f, Sxx, shading='gouraud')
            ax[y][x].set_ylabel('Frequency [Hz]')
            ax[y][x].set_xlabel('Time [sec]')
plt.savefig(fig)

WV_and_specgram(chirp, "WV - Base")
WV_and_specgram(chirp200, "WV - 200Hz")
WV_and_specgram(chirp300, "WV - 300Hz")
WV_and_specgram(chirp400, "WV - 400Hz")
WV_and_specgram(chirp500, "WV - 500Hz")


WV_and_specgram(chirp_sum_par, "WV - Paralel sum")
WV_and_specgram(chirp_sum_nonpar, "WV - non-paralel sum")

plt.show()
