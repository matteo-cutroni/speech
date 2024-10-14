#!/usr/bin/env python3
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

def load_audio(path):
    waveform, sr = torchaudio.load(path)
    return waveform, sr    

def mel_wave(waveform, sr):
    hop_length = int(sr/(1000/10))
    win_length = int(sr/(1000/25))
    tranform = torchaudio.transforms.MelSpectrogram(sr, n_mels=80, n_fft=win_length, hop_length=hop_length, win_length=win_length)
    mel_spec = tranform(waveform)
    return mel_spec[0].T

if __name__ == "__main__":
    waveform, sr = load_audio("LJSpeech-1.1/wavs/LJ001-0001.wav")
    mel_spec = mel_wave(waveform, sr)
    plt.imshow(np.log10(mel_spec.T))
    plt.show()
    plt.plot(waveform[0])
    plt.show()
