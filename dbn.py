import torch
from rbm import init_RBM, train_RBM, entree_sortie_RBM, sortie_entree_RBM, device


def init_DBN(layer_sizes):
    return [init_RBM(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]


def train_DBN(dbn, X, epochs=100, lr=0.1, batch_size=128):
    current = X.to(device)
    for i, rbm in enumerate(dbn):
        print(f"\n--- Training RBM layer {i+1}/{len(dbn)} ---")
        dbn[i] = train_RBM(rbm, current, epochs, lr, batch_size)
        with torch.no_grad():
            current = entree_sortie_RBM(dbn[i], current)
    return dbn


def generer_image_DBN(dbn, n_gibbs=1000, n_images=10):
    last = dbn[-1]
    p = last['b'].shape[0]
    q = last['a'].shape[0]

    h = (torch.rand(n_images, p, device=device) < 0.5).float()
    for _ in range(n_gibbs):
        pv = sortie_entree_RBM(last, h)
        v = (torch.rand_like(pv) < pv).float()
        ph = entree_sortie_RBM(last, v)
        h = (torch.rand_like(ph) < ph).float()

    current = v
    for rbm in reversed(dbn[:-1]):
        pv = sortie_entree_RBM(rbm, current)
        current = (torch.rand_like(pv) < pv).float()
    return current.cpu()
