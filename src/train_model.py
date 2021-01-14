import torch
import numpy as np
from utils import one_hot_encode, pad_char_sequences
import time

models_dir = '../saved_models/'

if not torch.cuda.is_available():
    dev = "cpu"
else:
    dev = "cuda:0"
device = torch.device(dev)

torch.manual_seed(0)
np.random.seed(0)


def train_model(model, vocab, train_dl, learning_rate=0.003, num_epochs=5):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_dl)
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_dl):
            sequences, lengths, functions = batch['sequence'], batch['length'], batch['function']
            sequences = pad_char_sequences(sequences)
            sequences = one_hot_encode(sequences, vocab).float()
            sequences, lengths, functions = sequences.to(device), lengths.cpu(), functions.to(device)

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
    torch.save(model.state_dict(), models_dir + 'model_{}.ckpt'.format(time.strftime("%Y%m%d-%H%M%S")))

    return model


