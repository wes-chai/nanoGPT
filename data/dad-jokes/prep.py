import os
import requests
import tiktoken
import numpy as np
import pandas as pd

# no download needed, it is in the folder
df = pd.read_csv('data/dad-jokes/dad-a-base.csv')
data = df['Joke'].str.cat(sep='\n')

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 14k tokens
# val.bin has 1.5k tokens

# Step 1: python data/dad-jokes/prep.py
# Step 2 - Train data: train_jokes.py