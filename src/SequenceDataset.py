import torch
import numpy as np

data_dir = '../data/processed/'
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class SequenceDataset(torch.utils.data.Dataset):
    """docstring for SequenceDataset"""

    def __init__(self):
        super(SequenceDataset, self).__init__()
        self.X = np.loadtxt(data_dir + 'sequence.txt', dtype=object)
        self.y = np.loadtxt(data_dir + 'function.txt', dtype=object)

        vocab = set()
        for seq in self.X:
            for c in seq:
                vocab.add(c)

        self.vocab = ['0'] + sorted(list(vocab))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequence = self.X[idx]
        function = torch.tensor(int(self.y[idx]))
        function = function.to(device)
        length = torch.tensor(len(sequence))
        function = function.to(device)

        sample = {'sequence': sequence, 'length': length, 'function': function}

        return sample

    def get_vocab(self):
        return self.vocab
