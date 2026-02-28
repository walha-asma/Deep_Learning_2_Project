import numpy as np
from rbm import init_RBM, train_RBM, entree_sortie_RBM, sortie_entree_RBM


def init_DBN(layer_sizes):
    return [init_RBM(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]


def train_DBN(dbn, X, epochs=100, lr=0.1, batch_size=128):
    current = X.copy()
    for i, rbm in enumerate(dbn):
        print(f"\n--- Training RBM layer {i+1}/{len(dbn)} ---")
        dbn[i] = train_RBM(rbm, current, epochs, lr, batch_size)
        current = entree_sortie_RBM(dbn[i], current)
    return dbn


def generer_image_DBN(dbn, n_gibbs=1000, n_images=10):
    last = dbn[-1]
    p = last['b'].shape[0]
    q = last['a'].shape[0]
    
    h = (np.random.rand(n_images, p) < 0.5).astype(float)
    for _ in range(n_gibbs):
        v = (np.random.rand(n_images, q) < sortie_entree_RBM(last, h)).astype(float)
        h = (np.random.rand(n_images, p) < entree_sortie_RBM(last, v)).astype(float)
    
    # Propagate backwards through all layers
    current = v
    for rbm in reversed(dbn[:-1]):
        current = (np.random.rand(n_images, rbm['a'].shape[0]) < sortie_entree_RBM(rbm, current)).astype(float)
    return current
