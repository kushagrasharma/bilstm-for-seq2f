import torch
import numpy as np
from utils import one_hot_encode, pad_char_sequences

models_dir = '../saved_models/'

if not torch.cuda.is_available():
    dev = "cpu"
else:
    dev = "cuda:0"
device = torch.device(dev)

torch.manual_seed(0)
np.random.seed(0)


def test_model(model, vocab, test_ds, test_dl) -> None:
    """

    @rtype: NoneType
    """
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_dl:
            sequences, lengths, functions = batch['sequence'], batch['length'], batch['function']
            sequences = pad_char_sequences(sequences)
            sequences = one_hot_encode(sequences, vocab)
            sequences, lengths, functions = sequences.to(device), lengths.to(device), functions.to(device)

            output = model(sequences, lengths)

            _, predicted = torch.max(output, 1)
            total += len(functions)
            correct += (predicted == functions).sum().item()

        print('Test Accuracy of the model on the {} test sequences: {} %'.format(len(test_ds), 100 * correct / total))
