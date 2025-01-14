import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import random

# Sample text for training
sample_text = """
Once upon a time, there was a small village surrounded by mountains.
The villagers lived happily, growing crops and raising livestock.
One day, a stranger arrived, bringing with him stories of far-off lands.
He spoke of treasures, dangers, and adventures beyond imagination.
The villagers were captivated by his tales, dreaming of what lay beyond the mountains.
"""

# Dataset class for text
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

# LSTM model with PyTorch Lightning
class LSTMModel(pl.LightningModule):
    def __init__(self, vocab_size, hidden_size=128, num_layers=2, seq_length=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

    def training_step(self, batch, batch_idx):
        input_seq, target_seq = batch
        logits, _ = self(input_seq)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_seq.view(-1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Function to generate new text
def generate_text(model, dataset, start_text="Once", length=100):
    model.eval()
    generated = list(start_text)
    input_seq = torch.tensor([dataset.char_to_idx[ch] for ch in start_text], dtype=torch.long).unsqueeze(0)

    hidden = None
    for _ in range(length):
        logits, hidden = model(input_seq, hidden)
        probs = torch.softmax(logits[:, -1, :], dim=-1).detach().squeeze().numpy()
        next_idx = random.choices(range(len(probs)), weights=probs)[0]
        next_char = dataset.idx_to_char[next_idx]
        generated.append(next_char)
        input_seq = torch.tensor([[next_idx]], dtype=torch.long)

    return "".join(generated)

# Main script
if __name__ == "__main__":
    # Dataset and DataLoader
    dataset = TextDataset(sample_text, seq_length=50)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model initialization
    model = LSTMModel(vocab_size=len(dataset.chars))

    # Training
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, dataloader)

    # Generate new text
    generated_text = generate_text(model, dataset, start_text="Once", length=200)
    print("\nGenerated Text:")
    print(generated_text)
