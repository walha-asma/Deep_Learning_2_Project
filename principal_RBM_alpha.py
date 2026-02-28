import numpy as np
import matplotlib.pyplot as plt
from data_utils import lire_alpha_digit
from rbm import init_RBM, train_RBM, generer_image_RBM

# ---- Parameters ----
ALPHA_PATH = 'data/binaryalphadigs.mat'
characters = [10, 11, 12]  # A, B, C
n_hidden = 200
epochs = 100
lr = 0.1
batch_size = 64
n_gibbs = 1000
n_images = 5

# ---- Load data ----
X = lire_alpha_digit(ALPHA_PATH, characters)
print(f"Data shape: {X.shape}")

# ---- Train RBM ----
rbm = init_RBM(X.shape[1], n_hidden)
rbm = train_RBM(rbm, X, epochs=epochs, lr=lr, batch_size=batch_size)

# ---- Generate images ----
generated = generer_image_RBM(rbm, n_gibbs=n_gibbs, n_images=n_images)

fig, axes = plt.subplots(1, n_images, figsize=(n_images * 2, 2))
for i, ax in enumerate(axes):
    ax.imshow(generated[i].reshape(20, 16), cmap='gray')
    ax.axis('off')
plt.suptitle('RBM Generated Images')
plt.tight_layout()
plt.savefig('outputs/rbm_generated.png')
plt.show()
print("Done.")
