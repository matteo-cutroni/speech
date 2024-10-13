#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

padded = np.load('ljs_tokenized.npy')

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

        self.embedding = nn.Embedding(vocab_size, d_model)
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
        self.conv1 = nn.Conv1d(d_model, conv_filter_size, kernel_size)
        self.ln = nn.LayerNorm(conv_filter_size)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(conv_filter_size, conv_filter_size, kernel_size)
        self.linear = nn.Linear(conv_filter_size, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.dropout(self.ln(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(self.ln(x))
        x = x.transpose(1, 2)
        return self.linear(x).squeeze(-1)
    
