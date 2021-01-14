import torch
import numpy as np
from BiLSTM import BiLSTM
from SequenceDataset import SequenceDataset
import glob
import os
from train_model import train_model
from test_model import test_model

if not torch.cuda.is_available():
    dev = "cpu"
else:
    dev = "cuda:0"
device = torch.device(dev)

models_dir = '../saved_models/'

torch.manual_seed(0)
np.random.seed(0)

batch_size = 2
num_epochs = 5
num_classes = 5
learning_rate = 0.003

model = BiLSTM(batch_size=batch_size)

list_of_files = glob.glob(models_dir + '*.ckpt')

if list_of_files:
    latest_file = max(list_of_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_file))

model.to(device)

ds = SequenceDataset()
vocab = ds.get_vocab()
train_len = int(len(ds) * 0.8)
test_len = len(ds) - train_len
train_ds, test_ds = torch.utils.data.random_split(ds, [train_len, test_len])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

model = train_model(model, vocab, train_dl, learning_rate, num_epochs)
test_model(model, vocab, test_ds, test_dl)
