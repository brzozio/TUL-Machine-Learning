import os
from main import TransformerDecoderModel
from main import TextDataset


START_TEXT = ["N","M","Litwo","Bigos"]
EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 264

EPOCHS = 20
SAVE_AFTER_EPOCH = 4

BATCH_SIZE = 128
SEQUENCE_LENGTH = 128

SRC_PATH : str  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH : str  = os.path.dirname(SRC_PATH) + '\\data'
CHCK_PATH : str  = os.path.dirname(SRC_PATH) + '\\models'
checkpoint_path = f"{CHCK_PATH}\\tranformer-epoch=19.ckpt"

file_path = f"{DATA_PATH}\\train.txt"
with open(file_path, "r", encoding="utf-8") as f:
    training_text = f.read()
dataset = TextDataset(training_text, seq_length=SEQUENCE_LENGTH)


model = TransformerDecoderModel.load_from_checkpoint(checkpoint_path, vocab_size=len(dataset.chars),
                                                    embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS,
                                                    num_layers=NUM_LAYERS,dim_feedforward=DIM_FEEDFORWARD,seq_len=SEQUENCE_LENGTH)
model.to("cuda")
model.eval()

for init in START_TEXT:
    print(f"\npoczątkowy kontekst:\"{init}\"\n\"{model.generate_text(start_text=init, max_length=100, 
                                start_tokens=[dataset.char_to_idx.get(char, 0) for char in init], id_to_char=dataset.idx_to_char)}\"")
    
