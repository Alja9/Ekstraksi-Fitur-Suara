import os
import time
import numpy as np
import pandas
import speechpy
import scipy
import sounddevice as sd
import scipy.fftpack as fft
from scipy.io.wavfile import write
from scipy.io import wavfile
from scipy.signal import get_window

def normalize_audio(audio):
    audio = audio/np.max(np.abs(audio))
    return audio

def frame_audio(audio, FFT_size = 2048, hop_size = 10, sample_rate=44100):
    audio = np.pad(audio, int(FFT_size/2), mode='reflect')
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num, FFT_size))
    for n in range(frame_num):
        frames[n] = audio[n*frame_len : n*frame_len+FFT_size]
    return frames

def make_matrix_X(x, p):
    n = len(x)
    xz = np.concatenate([x[::-1], np.zeros(p)])
    X = np.zeros((n - 1, p))
    for i in range(n - 1):
        offset = n - 1 - i 
        X[i, :] = xz[offset : offset + p]
    return X

def solve_lpc(x, p, ii):
    b = x[1:].T
    X = make_matrix_X(x, p)
    a = np.linalg.lstsq(X, b, rcond=None)[0]
    e = b.T - np.dot(X, a)
    g = np.var(e)
    return [a, g]

def rawRecord(fs=48000, s=3):
    print("--Start Record--")
    time.sleep(0.1)
    myrecording = sd.rec(int(s * fs), samplerate=fs, channels=2)
    sd.wait()
    write("rawRecord" + ".wav", fs, myrecording)
    time.sleep(0.1)
    print("--Finish Record--")
    time.sleep(0.6)
    
def convertLPC():
    rawRecord()
    fileNama = "rawRecord.wav"
    print("--Start--")
    time.sleep(0.1)
    print("Processing :",fileNama)

    fiturmean = np.empty((40, 1))
    sample_rate, audio = wavfile.read(fileNama)

    if (len(audio.shape) > 1):
        audio1 = normalize_audio(audio[:,0])
    else:
        audio1 = normalize_audio(audio)
        
    threshold=0.1
    awal = 0
    audiohasil = audio1
    for x in range (len(audio1)):
        if np.abs(audio1[x]) >= threshold:
            awal=x
            break
    audiohasil = audio1[awal:len(audio1)]

    for x in range (len(audiohasil)):
        if np.abs(audiohasil[x]) >=threshold:
            akhir=x
    audiohasil2=audiohasil[0:akhir]

    hop_size = 12 
    FFT_size = 2048
    audio_framed = frame_audio(audiohasil2, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)

    window = get_window("hanning", FFT_size, fftbins=True)
    audio_win = audio_framed * window 

    p = 40
    def lpc_encode(audiohasil2, p, audio_win):
        B = audio_win
        [nb, nw] = B.shape
        A = np.zeros((p, nb))
        G = np.zeros((1, nb))
        for i in range(nb):
            [a, g] = solve_lpc(B[i, :], p, i)
            A[:, i] = a
            G[:, i] = g
        return [A, G]

    xyz = lpc_encode(audiohasil2, p, audio_win)
    xyz0 = xyz[0]
    for xpos in range(len(xyz0)):
        sigmax = 0
        for xn in xyz0[xpos,:]:
            sigmax += xn
        fiturmean[xpos,0] = sigmax/len(np.transpose(xyz0))    
    
    time.sleep(0.1)
    print("--Done--")

    indextable = []
    for i in range(40):
        indextable.append("fitur" + str(i+1))

    df = pandas.DataFrame(np.transpose(fiturmean),columns=indextable)
    df.to_excel("coefficientsLPC.xlsx", index=False)

convertLPC()
