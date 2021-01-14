import numpy as np
import torch


def seq_to_one_hot(seq, vocab, charToIndex):
    # One hot encodes a single sequence
    vector = np.zeros((len(seq), len(vocab)), dtype=float)
    for i, char in enumerate(seq):
        vector[i, charToIndex[char]] = 1
    return vector


def one_hot_encode(sequences, vocab):
    """
    One-hot encodes a list of sequences
    """
    charToIndex = {c: i for i, c in enumerate(vocab)}

    one_hot_sequences = np.zeros((len(sequences), len(sequences[0]), len(vocab)), dtype=float)

    for i, seq in enumerate(sequences):
        for j, char in enumerate(seq):
            one_hot_sequences[i, j, charToIndex[char]] = 1
    # From an earlier age of greater enlightenment
        # one_hot_sequences.append(seq_to_one_hot(seq, vocab, charToIndex))
    # one_hot_sequences = np.concatenate(one_hot_sequences)

    return torch.tensor(one_hot_sequences)


def pad_char_sequences(sequences):
    """
    Pads a list of sequences with '0' to the length of the longest sequence
    """
    pad_len = max([len(x) for x in sequences])
    sequences = [seq.ljust(pad_len, '0') for seq in sequences]
    return sequences
