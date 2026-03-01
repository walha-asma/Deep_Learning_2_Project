import torch
import numpy as np
import scipy.io
import struct
import os


def lire_alpha_digit(path, characters):
    data = scipy.io.loadmat(path)['dat']
    samples = []
    for c in characters:
        if isinstance(c, str):
            idx = int(c) if c.isdigit() else ord(c.upper()) - ord('A') + 10
        else:
            idx = c
        for j in range(data.shape[0]):
            samples.append(data[j, idx].flatten())
    X = np.array(samples, dtype=np.float32)
    return torch.tensor(X)


def load_mnist_images(path):
    with open(path, 'rb') as f:
        _, n, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    return torch.tensor((images > 127).astype(np.float32))


def load_mnist_labels(path):
    with open(path, 'rb') as f:
        _, n = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    one_hot = np.zeros((n, 10), dtype=np.float32)
    one_hot[np.arange(n), labels] = 1
    return torch.tensor(one_hot)


def load_mnist(data_dir):
    X_train = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    Y_train = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    X_test = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    Y_test = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    return X_train, Y_train, X_test, Y_test
