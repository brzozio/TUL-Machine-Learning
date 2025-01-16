import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import random
import os

START_TEXT = ["N","M","Litwo","Bigos"]
HIDDEN_SIZE = 128
SEQUENCE_LENGTH = 128
NUM_LAYERS = 1

SRC_PATH : str  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH : str  = os.path.dirname(SRC_PATH) + '\\data'
CHCK_PATH : str  = os.path.dirname(SRC_PATH) + '\\models'

class TextDataset(Dataset):
    def __init__(self, text, seq_length=50):
        self.chars = sorted(set(text))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}
        self.text = text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        input_seq = self.text[idx: idx + self.seq_length]
        target_seq = self.text[idx + 1: idx + self.seq_length + 1]
        input_tensor = torch.tensor([self.char_to_idx[ch] for ch in input_seq], dtype=torch.long)
        target_tensor = torch.tensor([self.char_to_idx[ch] for ch in target_seq], dtype=torch.long)
        return input_tensor, target_tensor

class LSTMModel(pl.LightningModule):
    def __init__(self, vocab_size, hidden_size=128, num_layers=1, seq_length=50):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.linear(output)
        return logits, hidden

    def training_step(self, batch, batch_idx):
        input_seq, target_seq = batch
        logits, _ = self(input_seq)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_seq.view(-1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def generate_text(model, dataset, start_text="A", length=100):
    model.eval()
    generated = list(start_text)
    input_seq = torch.tensor([dataset.char_to_idx[ch] for ch in start_text], dtype=torch.long).unsqueeze(0).to(model.device)

    hidden = None
    for _ in range(length):
        logits, hidden = model(input_seq, hidden)
        probs = torch.softmax(logits[:, -1, :], dim=-1).detach().squeeze().cpu().numpy()
        next_idx = random.choices(range(len(probs)), weights=probs)[0]
        next_char = dataset.idx_to_char[next_idx]
        generated.append(next_char)
        input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(model.device)

    return "".join(generated)

if __name__ == "__main__":

    file_path = f"{DATA_PATH}\\train.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        training_text = f.read()

    dataset = TextDataset(training_text, seq_length=SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=11, persistent_workers=True)
    model = LSTMModel(vocab_size=len(dataset.chars), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=10,
        save_top_k=-1,
        dirpath=CHCK_PATH,
        filename="lstm-text-gen-{epoch:02d}"
    )

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(model, dataloader)

    print(f"ilość tokenów w kontekście:\t{SEQUENCE_LENGTH}")
    print(f"rozmiar ukrytej warstwy LSTM:\t{HIDDEN_SIZE}")
    print(f"ilość warstw LSTM:\t{NUM_LAYERS}")
    for init in START_TEXT:
        print(f"\npoczątkowy kontekst:\"{init}\"\n\"{generate_text(model, dataset, start_text=init)}\"")