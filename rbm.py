import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_RBM(p, q):
    return {
        'W': torch.randn(p, q, device=device) * 0.01,
        'a': torch.zeros(p, device=device),
        'b': torch.zeros(q, device=device)
    }


def sigmoid(x):
    return torch.sigmoid(x)


def entree_sortie_RBM(rbm, X):
    return sigmoid(X @ rbm['W'] + rbm['b'])


def sortie_entree_RBM(rbm, H):
    return sigmoid(H @ rbm['W'].T + rbm['a'])


def train_RBM(rbm, X, epochs=100, lr=0.1, batch_size=128):
    n = X.shape[0]
    for epoch in range(epochs):
        indices = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch = X[indices[start:start + batch_size]].to(device)

            # Positive phase
            ph = entree_sortie_RBM(rbm, batch)
            h0 = (torch.rand_like(ph) < ph).float()

            # Negative phase
            pv1 = sortie_entree_RBM(rbm, h0)
            v1 = (torch.rand_like(pv1) < pv1).float()
            ph1 = entree_sortie_RBM(rbm, v1)

            m = batch.shape[0]
            rbm['W'] += lr * (batch.T @ ph - v1.T @ ph1) / m
            rbm['a'] += lr * (batch - v1).mean(dim=0)
            rbm['b'] += lr * (ph - ph1).mean(dim=0)

        recon = sortie_entree_RBM(rbm, entree_sortie_RBM(rbm, X.to(device)))
        mse = ((X.to(device) - recon) ** 2).mean().item()
        print(f"Epoch {epoch+1}/{epochs} - MSE: {mse:.4f}")
    return rbm


def generer_image_RBM(rbm, n_gibbs=1000, n_images=10):
    p = rbm['a'].shape[0]
    v = (torch.rand(n_images, p, device=device) < 0.5).float()
    for _ in range(n_gibbs):
        ph = entree_sortie_RBM(rbm, v)
        h = (torch.rand_like(ph) < ph).float()
        pv = sortie_entree_RBM(rbm, h)
        v = (torch.rand_like(pv) < pv).float()
    return v.cpu()
