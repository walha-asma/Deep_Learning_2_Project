import numpy as np
import scipy.io
import struct
import os


def lire_alpha_digit(path, characters):
    """
    Load Binary AlphaDigits dataset.
    characters: list of indices (0-35) or characters to load
    Returns matrix (n_samples, n_pixels)
    """
    data = scipy.io.loadmat(path)['dat']
    # data shape: (20, 36) where each cell is a 20x16 image
    # Indices: 0-9 = digits, 10-35 = A-Z
    
    samples = []
    for c in characters:
        if isinstance(c, str):
            if c.isdigit():
                idx = int(c)
            else:
                idx = ord(c.upper()) - ord('A') + 10
        else:
            idx = c
        for j in range(data.shape[0]):
            img = data[j, idx]
            samples.append(img.flatten())
    return np.array(samples, dtype=float)


def load_mnist_images(path):
    with open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
    images = images.reshape(n, rows * cols)
    # Binarize
    return (images > 127).astype(float)


def load_mnist_labels(path):
    with open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    one_hot = np.zeros((n, 10))
    one_hot[np.arange(n), labels] = 1
    return one_hot


def load_mnist(data_dir):
    X_train = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    Y_train = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    X_test = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    Y_test = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    return X_train, Y_train, X_test, Y_test
