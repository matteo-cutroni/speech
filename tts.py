#!/usr/bin/env python3
import model
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm


def collate_fn(batch):
    texts, mels = zip(*batch)
    
    texts = pad_sequence(texts, batch_first=True, padding_value=3)
    mels = pad_sequence(mels, batch_first=True, padding_value=3)
    
    return texts, mels

class LJSpeechDataset(Dataset):
    def __init__(self, text_data, mel_data):
        self.text_data = text_data
        self.mel_data = mel_data

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.text_data[idx]
        mel = self.mel_data[idx]
        return torch.LongTensor(text), torch.FloatTensor(mel)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # #colab
    # text_path = '/content/drive/MyDrive/speech/ljs_tokenized.pt'
    # mels_path = '/content/drive/MyDrive/speech/ljs_mels.pt'

    text_path = 'ljs_tokenized.pt'
    mels_path = 'ljs_mels.pt'

    batch_size = 16
    epochs = 10
    lr = 1e-4


    text_data = torch.load(text_path)
    mel_data = torch.load(mels_path)
    dataset = LJSpeechDataset(text_data, mel_data)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)

    m = model.TtS(vocab_size=8000)
    m.to(device)

    criterion = nn.MSELoss()
    opt = torch.optim.Adam(m.parameters(), lr)

    for epoch in tqdm(range(epochs)):
        m.train()
 
        tot_loss = 0

        for batch in tqdm(dataloader):
            texts, mels = batch
            texts = texts.to(device)
            mels = mels.to(device)

            opt.zero_grad()
            pred = m(texts)
            
            # Ensure the sequence lengths match
            min_length = min(pred.size(1), mels.size(1))  # Find the minimum length

            # Truncate both to the minimum length
            pred = pred[:, :min_length, :]
            mels = mels[:, :min_length, :]


            loss = criterion(pred, mels)
            loss.backward()
            
            opt.step()

            tot_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {tot_loss / len(dataloader)}")

