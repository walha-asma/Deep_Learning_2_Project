import numpy as np


def init_RBM(p, q):
    return {
        'W': np.random.normal(0, 0.01, (p, q)),
        'a': np.zeros(p),
        'b': np.zeros(q)
    }


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def entree_sortie_RBM(rbm, X):
    return sigmoid(X @ rbm['W'] + rbm['b'])


def sortie_entree_RBM(rbm, H):
    return sigmoid(H @ rbm['W'].T + rbm['a'])


def train_RBM(rbm, X, epochs=100, lr=0.1, batch_size=128):
    n = X.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch = X[indices[start:start + batch_size]]
            
            # Positive phase
            ph = entree_sortie_RBM(rbm, batch)
            h0 = (np.random.rand(*ph.shape) < ph).astype(float)
            
            # Negative phase
            pv1 = sortie_entree_RBM(rbm, h0)
            v1 = (np.random.rand(*pv1.shape) < pv1).astype(float)
            ph1 = entree_sortie_RBM(rbm, v1)
            
            # Updates
            m = batch.shape[0]
            rbm['W'] += lr * (batch.T @ ph - v1.T @ ph1) / m
            rbm['a'] += lr * np.mean(batch - v1, axis=0)
            rbm['b'] += lr * np.mean(ph - ph1, axis=0)
        
        recon = sortie_entree_RBM(rbm, entree_sortie_RBM(rbm, X))
        mse = np.mean((X - recon) ** 2)
        print(f"Epoch {epoch+1}/{epochs} - MSE: {mse:.4f}")
    return rbm


def generer_image_RBM(rbm, n_gibbs=1000, n_images=10):
    p = rbm['a'].shape[0]
    v = (np.random.rand(n_images, p) < 0.5).astype(float)
    for _ in range(n_gibbs):
        h = (np.random.rand(n_images, rbm['b'].shape[0]) < entree_sortie_RBM(rbm, v)).astype(float)
        v = (np.random.rand(n_images, p) < sortie_entree_RBM(rbm, h)).astype(float)
    return v
