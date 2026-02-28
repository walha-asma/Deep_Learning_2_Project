import numpy as np
import matplotlib.pyplot as plt
import copy
from data_utils import load_mnist
from dnn import init_DNN, pretrain_DNN, retropropagation, test_DNN, entree_sortie_reseau

# ---- Parameters ----
MNIST_DIR = 'data/mnist'
n_classes = 10
epochs_rbm = 100
epochs_bp = 200
lr = 0.1
batch_size = 128

# ---- Load data ----
X_train, Y_train, X_test, Y_test = load_mnist(MNIST_DIR)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")


def run_experiment(X_tr, Y_tr, X_te, Y_te, layer_sizes, pretrain=True):
    dnn1 = init_DNN(layer_sizes)
    dnn2 = copy.deepcopy(dnn1)

    if pretrain:
        print("\n[Pretraining DNN1]")
        dnn1 = pretrain_DNN(dnn1, X_tr, epochs=epochs_rbm, lr=lr, batch_size=batch_size)

    print("\n[Training DNN1 (pretrained)]")
    dnn1 = retropropagation(dnn1, X_tr, Y_tr, epochs=epochs_bp, lr=lr, batch_size=batch_size)

    print("\n[Training DNN2 (random init)]")
    dnn2 = retropropagation(dnn2, X_tr, Y_tr, epochs=epochs_bp, lr=lr, batch_size=batch_size)

    err1_train = test_DNN(dnn1, X_tr, Y_tr)
    err1_test = test_DNN(dnn1, X_te, Y_te)
    err2_train = test_DNN(dnn2, X_tr, Y_tr)
    err2_test = test_DNN(dnn2, X_te, Y_te)

    return err1_train, err1_test, err2_train, err2_test


# ============================================================
# Fig 1: Error vs number of layers (fixed 200 neurons/layer)
# ============================================================
print("\n=== Fig 1: Error vs number of layers ===")
n_layers_list = [2, 3, 4, 5]
input_dim = X_train.shape[1]
results_layers = []

for n_l in n_layers_list:
    hidden = [200] * n_l
    sizes = [input_dim] + hidden + [n_classes]
    print(f"\nLayers: {n_l} hidden, sizes: {sizes}")
    res = run_experiment(X_train, Y_train, X_test, Y_test, sizes)
    results_layers.append(res)
    print(f"  Pretrained: train={res[0]:.4f}, test={res[1]:.4f}")
    print(f"  Random:     train={res[2]:.4f}, test={res[3]:.4f}")

res = np.array(results_layers)
plt.figure()
plt.plot(n_layers_list, res[:, 1], 'b-o', label='Pretrained (test)')
plt.plot(n_layers_list, res[:, 3], 'r-o', label='Random (test)')
plt.xlabel('Number of hidden layers')
plt.ylabel('Error rate')
plt.title('Error rate vs number of layers (200 neurons/layer)')
plt.legend()
plt.grid(True)
plt.savefig('outputs/fig1_layers.png')
plt.show()


# ============================================================
# Fig 2: Error vs number of neurons per layer (fixed 2 layers)
# ============================================================
print("\n=== Fig 2: Error vs neurons per layer ===")
neurons_list = [100, 200, 300, 500, 700]
results_neurons = []

for n_neu in neurons_list:
    sizes = [input_dim, n_neu, n_neu, n_classes]
    print(f"\nNeurons per layer: {n_neu}")
    res = run_experiment(X_train, Y_train, X_test, Y_test, sizes)
    results_neurons.append(res)
    print(f"  Pretrained: train={res[0]:.4f}, test={res[1]:.4f}")
    print(f"  Random:     train={res[2]:.4f}, test={res[3]:.4f}")

res = np.array(results_neurons)
plt.figure()
plt.plot(neurons_list, res[:, 1], 'b-o', label='Pretrained (test)')
plt.plot(neurons_list, res[:, 3], 'r-o', label='Random (test)')
plt.xlabel('Neurons per layer')
plt.ylabel('Error rate')
plt.title('Error rate vs neurons per layer (2 hidden layers)')
plt.legend()
plt.grid(True)
plt.savefig('outputs/fig2_neurons.png')
plt.show()


# ============================================================
# Fig 3: Error vs number of training samples (2 layers x 200)
# ============================================================
print("\n=== Fig 3: Error vs number of training samples ===")
n_train_list = [1000, 3000, 7000, 10000, 30000, 60000]
sizes = [input_dim, 200, 200, n_classes]
results_data = []

for n_tr in n_train_list:
    X_sub, Y_sub = X_train[:n_tr], Y_train[:n_tr]
    print(f"\nTraining samples: {n_tr}")
    res = run_experiment(X_sub, Y_sub, X_test, Y_test, sizes)
    results_data.append(res)
    print(f"  Pretrained: train={res[0]:.4f}, test={res[1]:.4f}")
    print(f"  Random:     train={res[2]:.4f}, test={res[3]:.4f}")

res = np.array(results_data)
plt.figure()
plt.plot(n_train_list, res[:, 1], 'b-o', label='Pretrained (test)')
plt.plot(n_train_list, res[:, 3], 'r-o', label='Random (test)')
plt.xlabel('Number of training samples')
plt.ylabel('Error rate')
plt.title('Error rate vs training set size (2 layers x 200)')
plt.legend()
plt.grid(True)
plt.savefig('outputs/fig3_data.png')
plt.show()

print("\nAll experiments done. Figures saved in outputs/")
