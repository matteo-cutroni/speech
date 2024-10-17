#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

 
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=3)
        self.positional_encoding = PositionalEncoding(d_model)          
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
    
    def forward(self, x):
        x = self.embedding(x) * np.sqrt(x.size(1))
        x = self.positional_encoding(x)
        return self.transformer_encoder(x)


class VariancePredictor(nn.Module):
    def __init__(self, d_model, conv_filter_size=256, kernel_size=3, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, conv_filter_size, kernel_size, padding=1)
        self.ln = nn.LayerNorm(conv_filter_size)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(conv_filter_size, conv_filter_size, kernel_size, padding=1)
        self.linear = nn.Linear(conv_filter_size, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(F.relu(self.ln(self.conv1(x).transpose(1, 2)))).transpose(1, 2)
        x = self.dropout(F.relu(self.ln(self.conv2(x).transpose(1, 2)))).transpose(1, 2)
        x = self.linear(x.transpose(1, 2)).squeeze(-1)
        return x


class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, durations):
        ret = []
        for i, seq in enumerate(x):
            expanded_seq = []
            for j, frame in enumerate(seq):
                expanded_seq += [frame] * max(1, int(durations[i][j].item()))
            ret.append(torch.stack(expanded_seq))
        return nn.utils.rnn.pad_sequence(ret, batch_first=True)


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layers)

    def forward(self, x):
        x = self.positional_encoding(x)
        return self.transformer_decoder(x)
    

class TtS(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_mels=80, nhead=4, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024):
        super().__init__()

        self.encoder = Encoder(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward)
        self.duration_predictor = VariancePredictor(d_model)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(d_model)
        self.energy_predictor = VariancePredictor(d_model)
        self.decoder = Decoder(d_model, nhead, num_decoder_layers, dim_feedforward)
        self.mel_linear = nn.Linear(d_model, n_mels)

    def forward(self, x):
        x = self.encoder(x)

        durations = self.duration_predictor(x)
        x = self.length_regulator(x, durations)
        
        pitch_out = self.pitch_predictor(x)
        energy_out = self.energy_predictor(x)
        x = x + pitch_out.unsqueeze(-1) + energy_out.unsqueeze(-1)
        x = self.decoder(x)
        return self.mel_linear(x)