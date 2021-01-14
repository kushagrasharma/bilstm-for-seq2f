import torch
import numpy as np
from BiLSTM import BiLSTM
from SequenceDataset import SequenceDataset
from utils import one_hot_encode, pad_char_sequences
import glob
import os
import time

models_dir = '../saved_models/'

if not torch.cuda.is_available():
    dev = "cpu"
else:
    dev = "cuda:0"
device = torch.device(dev)

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
train_len = int(len(ds) * 0.8)
test_len = len(ds) - train_len
train_ds, test_ds = torch.utils.data.random_split(ds, [train_len, test_len])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_dl)
for epoch in range(num_epochs):
    for i, batch in enumerate(train_dl):
        sequences, lengths, functions = batch['sequence'], batch['length'], batch['function']
        sequences = pad_char_sequences(sequences)
        sequences = one_hot_encode(sequences, ds.get_vocab()).float()
        sequences, lengths, functions = sequences.to(device), lengths.to(device), functions.to(device)

        output = model(sequences, lengths)
        loss = criterion(output, functions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        if (i + 1) % 1000 == 0:
            torch.save(model.state_dict(), models_dir + 'model_{}.ckpt'.format(time.strftime("%Y%m%d-%H%M%S")))
            print("Saved model state to disk")

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in test_dl:
        sequences, lengths, functions = batch['sequence'], batch['length'], batch['function']
        sequences = pad_char_sequences(sequences)
        sequences = one_hot_encode(sequences, ds.get_vocab())
        sequences, lengths, functions = sequences.to(device), lengths.to(device), functions.to(device)

        output = model(sequences, lengths)

        _, predicted = torch.max(output, 1)
        total += len(functions)
        correct += (predicted == functions).sum().item()

    print('Test Accuracy of the model on the {} test sequences: {} %'.format(len(test_ds), 100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), models_dir + 'model_{}.ckpt'.format(time.strftime("%Y%m%d-%H%M%S")))
