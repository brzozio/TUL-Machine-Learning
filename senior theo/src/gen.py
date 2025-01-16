import os
from main import LSTMModel
from main import generate_text
from main import TextDataset


START_TEXT = ["N","M","Litwo","Bigos"]
HIDDEN_SIZE = 128
SEQUENCE_LENGTH = 128
NUM_LAYERS = 1

SRC_PATH : str  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH : str  = os.path.dirname(SRC_PATH) + '\\data'
CHCK_PATH : str  = os.path.dirname(SRC_PATH) + '\\models'

file_path = f"{DATA_PATH}\\train.txt"
with open(file_path, "r", encoding="utf-8") as f:
    training_text = f.read()

dataset = TextDataset(training_text, seq_length=SEQUENCE_LENGTH)

checkpoint_path = f"{CHCK_PATH}\\lstm-text-gen-epoch=04.ckpt"

model = LSTMModel.load_from_checkpoint(checkpoint_path, vocab_size=len(dataset.chars))
model.eval()

for init in START_TEXT:
    print(f"\npoczątkowy kontekst:\"{init}\"\n\"{generate_text(model, dataset, start_text=init)}\"")
