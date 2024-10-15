#!/usr/bin/env python3
import csv
import sentencepiece as spm
import numpy as np
import torch


def prepare_data(metadata_path, txt_path):
    with open(metadata_path) as csvfile:
        with open(txt_path, 'w') as f:
            reader = csv.reader(csvfile, delimiter='|')
            for row in reader:
                f.write(f"{row[1]}\n")

def create_tokenizer(txt_path):
    spm.SentencePieceTrainer.train(input=txt_path, model_prefix='ljspeech_bpe', vocab_size=8000, character_coverage=1, model_type='bpe')
    sp = spm.SentencePieceProcessor(model_file='ljspeech_bpe.model')
    return sp

def tokenize_data(sp, txt_path):
    tokenized = []
    eos_id = sp.eos_id()
    with open(txt_path, 'r') as f:
        for row in f:
            tokenized.append(sp.encode(row, out_type=int) + [eos_id])
    return tokenized

def pad_data(tokenized, sp):
    max_len = max(len(seq) for seq in tokenized)
    padded = np.full((len(tokenized), max_len), [sp.pad_id()])
    for i, seq in enumerate(tokenized):
        padded[i][:len(seq)] = seq
    return padded


if __name__ == "__main__":
    metadata_path = 'LJSpeech-1.1/metadata.csv'
    txt_path = 'LJSpeech-1.1/ljspeech.txt'
    prepare_data(metadata_path, txt_path)
    sp = create_tokenizer(txt_path)
    tokenized = tokenize_data(sp, txt_path)
    padded = pad_data(tokenized, sp)

    torch.save(padded, 'ljs_tokenized.pt')

    #example
    text = 'Forza Roma!'
    tokens = sp.encode(text, out_type=str)
    print(tokens)
    untokend = sp.decode(tokens)
    print(untokend)