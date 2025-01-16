import os
from main import LSTMModel
from main import generate_text
from main import TextDataset

HIDDEN_SIZE = 64
SEQUENCE_LENGTH = 128
NUM_LAYERS = 4

SRC_PATH : str  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH : str  = os.path.dirname(SRC_PATH) + '\\data'
CHCK_PATH : str  = os.path.dirname(SRC_PATH) + '\\models'
checkpoint_path = f"{CHCK_PATH}\\lstm-text-gen-epoch=00.ckpt"

file_path = f"{DATA_PATH}\\test.txt"

with open(file_path, "r", encoding="utf-8") as f:
    test_text = f.read()

dataset = TextDataset(test_text, seq_length=SEQUENCE_LENGTH)
model = LSTMModel.load_from_checkpoint(checkpoint_path, vocab_size=len(dataset.chars))
model.eval()

acc = 0
total = 0
with open(f"{DATA_PATH}\\test.txt", 'r', encoding='utf-8') as file:
    content = file.read(100)

    while len(content) == 100:
        temp = generate_text(model, dataset, start_text=content[:-1], length=1)
        if temp[-1] == content[-1]: acc = acc+1
        total = total+1
        content = file.read(100)

print(acc/total)
