import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from pytorch_lightning.callbacks import ModelCheckpoint
import os

SRC_PATH : str  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH : str  = os.path.dirname(SRC_PATH) + '\\data'
CHCK_PATH : str  = os.path.dirname(SRC_PATH) + '\\models'

EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 264

EPOCHS = 20
SAVE_AFTER_EPOCH = 4

BATCH_SIZE = 128
SEQUENCE_LENGTH = 128
LEARNING_RATE = 0.001

class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        input_seq = self.text[idx: idx + self.seq_length]
        target_seq = self.text[idx + 1: idx + self.seq_length + 1]
        input_tensor = torch.tensor([self.char_to_idx[ch] for ch in input_seq], dtype=torch.long)
        target_tensor = torch.tensor([self.char_to_idx[ch] for ch in target_seq], dtype=torch.long)
        return input_tensor, target_tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerDecoderModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dim_feedforward, seq_len):
        super(TransformerDecoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, seq_len)
        decoder_layer = TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, src, tgt):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32))
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32))
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, src)
        output = self.fc_out(output)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x, x)
        loss = nn.CrossEntropyLoss()(output.view(-1, output.size(-1)), y.view(-1))
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x, x)
        loss = nn.CrossEntropyLoss()(output.view(-1, len(dataset.chars)), y.view(-1))
        preds = torch.argmax(output, dim=-1)
        correct = (preds == y).float().sum()
        total = y.numel()
        accuracy = correct / total
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def generate_text(self, start_text, max_length=50, temperature=1.0, start_tokens=None, id_to_char=None):
        self.eval()

        if start_tokens==None:
            start_tokens = [dataset.char_to_idx.get(char, 0) for char in start_text]
            id_to_char = dataset.idx_to_char

        generated_tokens = start_tokens.copy()

        with torch.no_grad():
            for _ in range(max_length):
                input_seq = torch.tensor(generated_tokens[-self.seq_len:], dtype=torch.long).unsqueeze(0).to(self.device)
                output = self(input_seq, input_seq)
                logits = output[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated_tokens.append(next_token)

        generated_text = ''.join([id_to_char.get(token, '?') for token in generated_tokens]) 
        return generated_text
    

if __name__ == "__main__":


    file_path = f"{DATA_PATH}\\train.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        training_text = f.read()
    dataset = TextDataset(training_text, seq_length=SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=11, persistent_workers=True)    

    
    file_path = f"{DATA_PATH}\\test.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        validation_text = f.read()        
    validation_dataset = TextDataset(validation_text, seq_length=SEQUENCE_LENGTH)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=11, persistent_workers=True)

    model = TransformerDecoderModel(vocab_size=len(dataset.chars), embedding_dim=EMBEDDING_DIM, 
                                    num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD, 
                                    seq_len=SEQUENCE_LENGTH)

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=SAVE_AFTER_EPOCH,
        save_top_k=-1,
        dirpath=CHCK_PATH,
        filename="tranformer-{epoch:02d}"
    )

    trainer = Trainer(
        max_epochs=EPOCHS, 
        accelerator="gpu", 
        val_check_interval=0.2,
        callbacks=[checkpoint_callback]
        )
        

    trainer.fit(model, dataloader, validation_dataloader)

    start_text = "A"
    generated_text = model.generate_text(start_text, max_length=100)
    print("Generated Text:")
    print(generated_text)