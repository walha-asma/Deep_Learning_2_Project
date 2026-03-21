"""
Bonus: VAE on MNIST
Train and compare between:
  1. RBM
  2. DBN  (2 layers)
  3. VAE  (Variational Autoencoder)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os

from data_utils import load_mnist
from rbm import init_RBM, train_RBM, generer_image_RBM, device
from dbn import init_DBN, train_DBN, generer_image_DBN

MNIST_DIR = 'data/minst'
OUT_DIR   = 'outputs/bonus'
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES    = 10
N_GIBBS     = 500
EPOCHS_RBM  = 100
EPOCHS_VAE  = 30      
LR          = 0.001   # Adam LR for VAE
BATCH_SIZE  = 128
LATENT_DIM  = 64      # VAE latent space size
HIDDEN_DIM  = 256     # VAE encoder/decoder hidden size

print(f"Device: {device}")

# Load data (use subset for speed, full for best quality)  
print("Loading MNIST...")
X_train, _, X_test, _ = load_mnist(MNIST_DIR)
# Use 10k for RBM/DBN training  and all for VAE
X_rbm = X_train[:10000]
print(f"  RBM/DBN training set: {X_rbm.shape}")
print(f"  VAE training set:     {X_train.shape}")

# 1. RBM
print("\n Training RBM ")
rbm = init_RBM(784, 256)
rbm = train_RBM(rbm, X_rbm, epochs=EPOCHS_RBM, lr=0.1, batch_size=BATCH_SIZE)
rbm_images = generer_image_RBM(rbm, n_gibbs=N_GIBBS, n_images=N_IMAGES)

# 2. DBN (2 layers: 784->256->128)
print("\n Training DBN ")
dbn = init_DBN([784, 256, 128])
dbn = train_DBN(dbn, X_rbm, epochs=EPOCHS_RBM, lr=0.1, batch_size=BATCH_SIZE)
dbn_images = generer_image_DBN(dbn, n_gibbs=N_GIBBS, n_images=N_IMAGES)

# 3.  VAE
print("\n Training VAE ")

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        # Encoder
        self.enc_fc1  = nn.Linear(input_dim, hidden_dim)
        self.enc_fc2  = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu    = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.dec_fc1  = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc2  = nn.Linear(hidden_dim, hidden_dim)
        self.dec_out  = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.enc_fc1(x))
        h = torch.relu(self.enc_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.dec_fc1(z))
        h = torch.relu(self.dec_fc2(h))
        return torch.sigmoid(self.dec_out(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    # Binary cross-entropy reconstruction loss
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / x.shape[0]


vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=LR)

# Count params for report
n_params = sum(p.numel() for p in vae.parameters())
print(f"  VAE parameters: {n_params:,}")

train_losses = []
n = X_train.shape[0]

for epoch in range(EPOCHS_VAE):
    vae.train()
    indices  = torch.randperm(n)
    epoch_loss = 0.0
    for start in range(0, n, BATCH_SIZE):
        idx   = indices[start:start + BATCH_SIZE]
        batch = X_train[idx].to(device)
        optimizer.zero_grad()
        recon, mu, logvar = vae(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.shape[0]
    avg_loss = epoch_loss / n
    train_losses.append(avg_loss)
    print(f"  Epoch {epoch+1}/{EPOCHS_VAE} - ELBO loss: {avg_loss:.2f}")

# Generate VAE samples (sample z ~ N(0,1), decode)
vae.eval()
with torch.no_grad():
    z = torch.randn(N_IMAGES, LATENT_DIM, device=device)
    vae_samples = vae.decode(z).cpu()
    # Binarize for fair visual comparison
    vae_images = (vae_samples > 0.5).float()

vae_soft = vae_samples

# Save training curve for VAE 
plt.figure(figsize=(6, 4))
plt.plot(range(1, EPOCHS_VAE + 1), train_losses, 'purple')
plt.xlabel('Epoch')
plt.ylabel('ELBO loss (lower = better)')
plt.title('VAE training loss on MNIST')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/vae_training_loss.png', dpi=100)
plt.close()


 
# Final comparison figure: RBM vs DBN vs VAE
fig, axes = plt.subplots(3, N_IMAGES, figsize=(N_IMAGES * 1.5, 5))
model_names  = ['RBM', 'DBN (2 layers)', 'VAE']
image_groups = [rbm_images, dbn_images, vae_soft]

for row, (name, imgs) in enumerate(zip(model_names, image_groups)):
    for col in range(N_IMAGES):
        ax = axes[row, col]
        ax.imshow(imgs[col].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    axes[row, 0].set_ylabel(name, fontsize=11, rotation=90,
                             labelpad=8, va='center')

plt.suptitle('Generative model comparison on MNIST\n(RBM vs DBN vs VAE)',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/comparison_rbm_dbn_vae.png',
            dpi=120, bbox_inches='tight')
plt.close()
print(f"\nComparison figure saved: {OUT_DIR}/comparison_rbm_dbn_vae.png")

# Individual grids for each model 
def save_grid(images, title, path, shape=(28, 28)):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.5, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(shape), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

save_grid(rbm_images, 'RBM generated digits (MNIST)', f'{OUT_DIR}/rbm_mnist.png')
save_grid(dbn_images, 'DBN generated digits (MNIST)', f'{OUT_DIR}/dbn_mnist.png')
save_grid(vae_soft,   'VAE generated digits (MNIST)', f'{OUT_DIR}/vae_mnist.png')

# Save binarized VAE too
save_grid(vae_images, 'VAE generated digits — binarized (MNIST)',
          f'{OUT_DIR}/vae_mnist_binary.png')


# VAE interpolation: latent space walk
# Pick two random points in latent space and interpolate
print("\nGenerating VAE latent interpolation...")
vae.eval()
with torch.no_grad():
    z1 = torch.randn(1, LATENT_DIM, device=device)
    z2 = torch.randn(1, LATENT_DIM, device=device)
    steps = 10
    alphas = torch.linspace(0, 1, steps, device=device)
    interps = [vae.decode(z1 * (1 - a) + z2 * a).cpu() for a in alphas]

fig, axes = plt.subplots(1, steps, figsize=(steps * 1.5, 2))
for i, (ax, img) in enumerate(zip(axes, interps)):
    ax.imshow(img.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
plt.suptitle('VAE latent space interpolation', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/vae_interpolation.png', dpi=100, bbox_inches='tight')
plt.close()
print(f"  Saved {OUT_DIR}/vae_interpolation.png")

print(f"\nAll bonus outputs saved to: {OUT_DIR}/")
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
