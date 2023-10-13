import os
import requests
import numpy as np

# download the tiny shakespeare dataset
file_name = "suc_ve_ceza_fyodor_mihailovic_dostoyevski.txt"
input_file_path = os.path.join(os.path.dirname(__file__), file_name)
print(input_file_path)
with open(input_file_path, 'r',encoding="utf8") as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]


# encode with Turkish Bert
from transformers import BertTokenizer, BertForQuestionAnswering
tokenizer_tr = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')

train_ids = tokenizer_tr.encode_plus(train_data).input_ids
val_ids = tokenizer_tr.encode_plus(val_data).input_ids
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
