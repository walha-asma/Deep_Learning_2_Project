"""
AlphaDigits hyperparameter study.
Covers the 4 analyses required by Section 4 of the assignment:
  1. RBM — effect of number of hidden units
  2. RBM — effect of number of epochs
  3. RBM — effect of number of classes
  4. DBN — effect of depth (number of RBM layers)

All figures saved to outputs/alpha_study/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from data_utils import lire_alpha_digit
from rbm import init_RBM, train_RBM, generer_image_RBM
from dbn import init_DBN, train_DBN, generer_image_DBN

ALPHA_PATH = 'data/binaryalphadigs.mat'
OUT_DIR    = 'outputs/alpha_study'
os.makedirs(OUT_DIR, exist_ok=True)

N_GIBBS   = 500   # Gibbs steps for generation (kept moderate to save time)
N_IMAGES  = 5     # images to display per config
BASE_LR   = 0.1
BASE_BS   = 64

def save_image_grid(images, title, path, shape=(20, 16)):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2.4))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(shape), cmap='gray')
        ax.axis('off')
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────
# Helper: compute reconstruction MSE at epoch 0
# (we just use a short train and read last MSE)
# ─────────────────────────────────────────────
def train_and_get_mse(rbm, X, epochs, lr, batch_size):
    """Returns (trained_rbm, final_mse)."""
    import torch
    from rbm import device, entree_sortie_RBM, sortie_entree_RBM

    n = X.shape[0]
    final_mse = None
    for epoch in range(epochs):
        indices = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch = X[indices[start:start + batch_size]].to(device)
            ph  = entree_sortie_RBM(rbm, batch)
            h0  = (torch.rand_like(ph) < ph).float()
            pv1 = sortie_entree_RBM(rbm, h0)
            v1  = (torch.rand_like(pv1) < pv1).float()
            ph1 = entree_sortie_RBM(rbm, v1)
            m   = batch.shape[0]
            rbm['W'] += lr * (batch.T @ ph - v1.T @ ph1) / m
            rbm['a'] += lr * (batch - v1).mean(dim=0)
            rbm['b'] += lr * (ph - ph1).mean(dim=0)

        recon = sortie_entree_RBM(rbm, entree_sortie_RBM(rbm, X.to(device)))
        final_mse = ((X.to(device) - recon) ** 2).mean().item()

    return rbm, final_mse


# ══════════════════════════════════════════════
# STUDY 1 — RBM: effect of hidden units
# Characters: A, B, C  |  Epochs: 100
# Hidden units: 50, 100, 200, 500
# ══════════════════════════════════════════════
print("\n=== Study 1: RBM hidden units ===")
chars      = [10, 11, 12]   # A, B, C
epochs_s1  = 100
units_list = [50, 100, 200, 500]
mse_units  = []

X = lire_alpha_digit(ALPHA_PATH, chars)

for q in units_list:
    print(f"  q={q} ...", end=" ", flush=True)
    rbm = init_RBM(X.shape[1], q)
    rbm, mse = train_and_get_mse(rbm, X, epochs_s1, BASE_LR, BASE_BS)
    mse_units.append(mse)
    imgs = generer_image_RBM(rbm, n_gibbs=N_GIBBS, n_images=N_IMAGES)
    save_image_grid(imgs,
                    f'RBM  q={q}  MSE={mse:.4f}',
                    f'{OUT_DIR}/rbm_hidden_q{q}.png')

# Summary plot
plt.figure(figsize=(6, 4))
plt.plot(units_list, mse_units, 'b-o')
plt.xlabel('Number of hidden units')
plt.ylabel('Reconstruction MSE')
plt.title('RBM: reconstruction MSE vs hidden units')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/rbm_hidden_mse.png', dpi=100)
plt.close()
print("  Summary saved.")


# ══════════════════════════════════════════════
# STUDY 2 — RBM: effect of number of epochs
# Characters: A, B, C  |  Hidden: 200
# Epochs: 50, 100, 200, 500
# ══════════════════════════════════════════════
print("\n=== Study 2: RBM epochs ===")
epoch_list = [50, 100, 200, 500]
mse_epochs = []

for ep in epoch_list:
    print(f"  epochs={ep} ...", end=" ", flush=True)
    rbm = init_RBM(X.shape[1], 200)
    rbm, mse = train_and_get_mse(rbm, X, ep, BASE_LR, BASE_BS)
    mse_epochs.append(mse)
    imgs = generer_image_RBM(rbm, n_gibbs=N_GIBBS, n_images=N_IMAGES)
    save_image_grid(imgs,
                    f'RBM  epochs={ep}  MSE={mse:.4f}',
                    f'{OUT_DIR}/rbm_epochs_{ep}.png')

plt.figure(figsize=(6, 4))
plt.plot(epoch_list, mse_epochs, 'g-o')
plt.xlabel('Number of epochs')
plt.ylabel('Reconstruction MSE')
plt.title('RBM: reconstruction MSE vs epochs')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/rbm_epochs_mse.png', dpi=100)
plt.close()
print("  Summary saved.")


# ══════════════════════════════════════════════
# STUDY 3 — RBM: effect of number of classes
# Hidden: 200  |  Epochs: 100
# Classes: 1, 3, 6, 10
# ══════════════════════════════════════════════
print("\n=== Study 3: RBM number of classes ===")
class_configs = [
    ([10],                        "1 class (A)"),
    ([10, 11, 12],                "3 classes (A,B,C)"),
    ([10, 11, 12, 13, 14, 15],    "6 classes (A–F)"),
    (list(range(10, 20)),         "10 classes (A–J)"),
]
mse_classes  = []
class_labels = []

for char_list, label in class_configs:
    print(f"  {label} ...", end=" ", flush=True)
    Xc = lire_alpha_digit(ALPHA_PATH, char_list)
    rbm = init_RBM(Xc.shape[1], 200)
    rbm, mse = train_and_get_mse(rbm, Xc, 100, BASE_LR, BASE_BS)
    mse_classes.append(mse)
    class_labels.append(len(char_list))
    imgs = generer_image_RBM(rbm, n_gibbs=N_GIBBS, n_images=N_IMAGES)
    save_image_grid(imgs,
                    f'RBM  {label}  MSE={mse:.4f}',
                    f'{OUT_DIR}/rbm_classes_{len(char_list)}.png')

plt.figure(figsize=(6, 4))
plt.plot(class_labels, mse_classes, 'r-o')
plt.xlabel('Number of character classes')
plt.ylabel('Reconstruction MSE')
plt.title('RBM: reconstruction MSE vs number of classes')
plt.xticks(class_labels)
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/rbm_classes_mse.png', dpi=100)
plt.close()
print("  Summary saved.")


# ══════════════════════════════════════════════
# STUDY 4 — DBN: effect of depth
# Characters: A, B, C  |  Hidden: 200 per layer
# Depths: 1, 2, 3, 4 RBM layers
# ══════════════════════════════════════════════
print("\n=== Study 4: DBN depth ===")
X = lire_alpha_digit(ALPHA_PATH, [10, 11, 12])
depth_configs = [
    [320, 200],
    [320, 200, 200],
    [320, 200, 200, 200],
    [320, 200, 200, 200, 200],
]

for sizes in depth_configs:
    depth = len(sizes) - 1
    label = f'{depth} RBM layer{"s" if depth > 1 else ""}'
    print(f"  {label} ...", end=" ", flush=True)
    dbn = init_DBN(sizes)
    dbn = train_DBN(dbn, X, epochs=100, lr=BASE_LR, batch_size=BASE_BS)
    imgs = generer_image_DBN(dbn, n_gibbs=N_GIBBS, n_images=N_IMAGES)
    save_image_grid(imgs,
                    f'DBN  {label}  arch={sizes}',
                    f'{OUT_DIR}/dbn_depth_{depth}.png')

print("\nAll studies complete. Results in:", OUT_DIR)
