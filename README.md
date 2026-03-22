# Projet Deep Learning II — RBM, DBN, DNN & VAE

**Institut Polytechnique de Paris — ENSTA Paris**

Implémentation from scratch en PyTorch de Restricted Boltzmann Machines (RBM), Deep Belief Networks (DBN), Deep Neural Networks (DNN) pré-entraînés, et d'un Variational Autoencoder (VAE) pour la comparaison.

---

## Structure du projet

```
TP_DNN/
│
├── data/
│   ├── binaryalphadigs.mat       ← Binary AlphaDigits dataset
│   └── minst/
│       ├── train-images-idx3-ubyte
│       ├── train-labels-idx1-ubyte
│       ├── t10k-images-idx3-ubyte
│       └── t10k-labels-idx1-ubyte
│
├── rbm.py                            ← RBM : init, train (CD-1), génération
├── dbn.py                            ← DBN : init, train (greedy layer-wise), génération
├── dnn.py                            ← DNN : init, pré-entraînement, rétropropagation, test
├── data_utils.py                     ← Chargement AlphaDigits et MNIST
│
├── principal_RBM_alpha.py            ← Script RBM sur AlphaDigits (run rapide)
├── principal_DBN_alpha.py            ← Script DBN sur AlphaDigits (run rapide)
├── principal_alpha_study.py          ← Étude hyperparamètres complète sur AlphaDigits
├── principal_DNN_MNIST.py            ← Étude DNN pré-entraîné vs aléatoire sur MNIST
├── principal_bonus_VAE.py            ← Bonus : comparaison RBM / DBN / VAE sur MNIST
│
├── jobs/
│   ├── run_rbm_alpha.sh
│   ├── run_dbn_alpha.sh
│   ├── run_alpha_study.sh
│   ├── run_dnn_mnist.sh
│   └── run_bonus_vae.sh
│
├── outputs/                          ← Figures générées (créé automatiquement)
│   ├── alpha_study/
│   └── bonus/
│
├── requirements.txt
└── README.md
```

---

## Données

### Binary AlphaDigits
Télécharger depuis : https://www.kaggle.com/datasets/angevalli/binary-alpha-digits

Placer le fichier `binaryalphadigs.mat` dans `data/alpha/`.

### MNIST
Télécharger les 4 fichiers depuis : http://yann.lecun.com/exdb/mnist/

```
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
```

Décompresser et placer dans `data/minst/`.

---

## Installation

### Environnement virtuel dédié

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Exécution locale

```bash
# Vérification rapide RBM sur AlphaDigits 
python principal_RBM_alpha.py

# Vérification rapide DBN sur AlphaDigits 
python principal_DBN_alpha.py

# Étude hyperparamètres AlphaDigits complète 
python principal_alpha_study.py

# Étude DNN sur MNIST — figures principales 
python principal_DNN_MNIST.py

# Bonus VAE — comparaison RBM/DBN/VAE 
python principal_bonus_VAE.py
```

---

## Exécution sur cluster SLURM

```bash
# Créer les dossiers nécessaires
mkdir -p jobs/logs outputs/alpha_study outputs/bonus

# Soumettre les jobs (les 3 derniers peuvent tourner en parallèle)
sbatch jobs/run_rbm_alpha.sh
sbatch jobs/run_dbn_alpha.sh
sbatch jobs/run_alpha_study.sh
sbatch jobs/run_dnn_mnist.sh
sbatch jobs/run_bonus_vae.sh

# Suivre l'avancement
squeue -u $USER
tail -f jobs/logs/dnn_mnist_<JOB_ID>.out
```

---

## Description des modules

### `rbm.py`
- `init_RBM(p, q)` — initialise W ~ N(0, 0.01), a=0, b=0
- `entree_sortie_RBM(rbm, X)` — passage visible → caché (sigmoïde)
- `sortie_entree_RBM(rbm, H)` — passage caché → visible (sigmoïde)
- `train_RBM(rbm, X, epochs, lr, batch_size)` — entraînement CD-1, affiche MSE à chaque epoch
- `generer_image_RBM(rbm, n_gibbs, n_images)` — échantillonnage de Gibbs

### `dbn.py`
- `init_DBN(layer_sizes)` — initialise une liste de RBMs
- `train_DBN(dbn, X, epochs, lr, batch_size)` — greedy layer-wise pretraining
- `generer_image_DBN(dbn, n_gibbs, n_images)` — Gibbs sur le dernier RBM puis propagation inverse

### `dnn.py`
- `init_DNN(layer_sizes)` — identique à init_DBN (DNN = DBN + couche classification)
- `pretrain_DNN(dnn, X, ...)` — pré-entraîne toutes les couches sauf la dernière via train_DBN
- `calcul_softmax(rbm, X)` — softmax numérique stable pour la couche de sortie
- `entree_sortie_reseau(dnn, X)` — forward pass complet, retourne les activations de chaque couche
- `retropropagation(dnn, X, Y, ...)` — descente de gradient, retourne `(dnn, loss_history)`
- `test_DNN(dnn, X, Y)` — calcule le taux d'erreur

### `data_utils.py`
- `lire_alpha_digit(path, characters)` — charge les caractères sélectionnés depuis le `.mat`
- `load_mnist(data_dir)` — charge et binarise MNIST (seuil 127), retourne X_train, Y_train, X_test, Y_test en one-hot

---

## Résultats obtenus

### AlphaDigits — Étude hyperparamètres

| Paramètre | Tendance observée |
|-----------|-------------------|
| Unités cachées (50→500) | MSE diminue monotoniquement, rendements décroissants après 200 |
| Epochs (50→500) | MSE chute fortement jusqu'à 200 epochs puis se stabilise |
| Nombre de classes (1→10) | Pic à 3 classes (complexité > données), puis baisse (plus de données) |
| Profondeur DBN (1→4 couches) | Amélioration progressive, chaque couche affine la représentation |

### MNIST — DNN pré-entraîné vs aléatoire

| Figure | Résultat clé |
|--------|-------------|
| Fig 1 (couches) | Le réseau aléatoire s'effondre à 88% d'erreur dès 5 couches (gradient vanishing). Le pré-entraîné reste stable à ~2% |
| Fig 2 (neurones) | Les deux s'améliorent avec plus de neurones, le pré-entraîné garde un avantage constant (~0.5%) |
| Fig 3 (données) | L'écart est maximal avec peu de données (11% vs 90% à 1000 samples) et se réduit à 60k samples |

Meilleure configuration trouvée : **2 couches cachées × 500 neurones, pré-entraîné**, taux d'erreur ~1.9% sur le jeu de test.

### Bonus — Comparaison RBM / DBN / VAE

| Modèle | Qualité visuelle | Diversité | Interpolation latente |
|--------|-----------------|-----------|----------------------|
| RBM | Blobs bruités, formes reconnaissables | Faible | Non |
| DBN | Formes plus nettes, meilleure cohérence | Modérée | Non |
| VAE | Digits nets, variés, réalistes | Élevée | ✓ (2→7 continu) |

---

## Dépendances

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
```

---

## Notes techniques

- Tout le code utilise automatiquement le GPU si disponible (`cuda`) via `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Le MNIST est binarisé avec un seuil à 127 (pixel > 127 → 1, sinon → 0)
- La rétropropagation utilise softmax + cross-entropie pour la couche de sortie et sigmoid pour les couches cachées
- Le VAE utilise l'optimiseur Adam (lr=0.001) et la loss ELBO (reconstruction BCE + KL divergence)
