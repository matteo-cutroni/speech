#!/usr/bin/env python3
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def load_audio(path):
    waveform, sr = torchaudio.load(path)
    return waveform, sr    

def mel_wave(waveform, sr):
    hop_length = int(sr/(1000/10))
    win_length = int(sr/(1000/25))
    tranform = torchaudio.transforms.MelSpectrogram(sr, n_mels=80, n_fft=win_length, hop_length=hop_length, win_length=win_length)
    mel_spec = tranform(waveform)
    return mel_spec[0].T

def create_mels(wavs_folder):
    mel_data = []
    for file in tqdm(os.listdir(wavs_folder)):
        file_path = os.path.join(wavs_folder, file)
        waveform, sr = load_audio(file_path)
        mel_spec = mel_wave(waveform, sr)
        mel_data.append(mel_spec)

    return mel_data



if __name__ == "__main__":
    
    wavs_folder = "LJSpeech-1.1/wavs"
    mel_data = create_mels(wavs_folder)
    torch.save(mel_data, 'ljs_mels.pt')
    
    
    #example
    waveform, sr = load_audio("LJSpeech-1.1/wavs/LJ001-0001.wav")
    mel_spec = mel_wave(waveform, sr)
    plt.imshow(np.log10(mel_spec.T))
    plt.show()
    plt.plot(waveform[0])
    plt.show()
