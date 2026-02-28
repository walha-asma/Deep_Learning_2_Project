import numpy as np
import matplotlib.pyplot as plt
from data_utils import lire_alpha_digit
from dbn import init_DBN, train_DBN, generer_image_DBN

# ---- Parameters ----
ALPHA_PATH = 'data/binaryalphadigs.mat'
characters = [10, 11, 12]  # A, B, C
layer_sizes = [320, 200, 100]  # input (20x16=320), hidden layers
epochs = 100
lr = 0.1
batch_size = 64
n_gibbs = 1000
n_images = 5

# ---- Load data ----
X = lire_alpha_digit(ALPHA_PATH, characters)
print(f"Data shape: {X.shape}")

# ---- Train DBN ----
dbn = init_DBN(layer_sizes)
dbn = train_DBN(dbn, X, epochs=epochs, lr=lr, batch_size=batch_size)

# ---- Generate images ----
generated = generer_image_DBN(dbn, n_gibbs=n_gibbs, n_images=n_images)

fig, axes = plt.subplots(1, n_images, figsize=(n_images * 2, 2))
for i, ax in enumerate(axes):
    ax.imshow(generated[i].reshape(20, 16), cmap='gray')
    ax.axis('off')
plt.suptitle('DBN Generated Images')
plt.tight_layout()
plt.savefig('outputs/dbn_generated.png')
plt.show()
print("Done.")
