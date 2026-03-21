import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import copy
import os
from data_utils import load_mnist
from dnn import init_DNN, pretrain_DNN, retropropagation, test_DNN
from rbm import device

MNIST_DIR = 'data/minst'
epochs_rbm = 100
epochs_bp = 200
lr = 0.1
batch_size = 128

os.makedirs('outputs', exist_ok=True)

print(f"Using device: {device}")
X_train, Y_train, X_test, Y_test = load_mnist(MNIST_DIR)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
input_dim = X_train.shape[1]
n_classes = 10


def run_experiment(X_tr, Y_tr, X_te, Y_te, layer_sizes):
    dnn1 = init_DNN(layer_sizes)
    dnn2 = copy.deepcopy(dnn1)

    print("\n[Pretraining DNN1]")
    dnn1 = pretrain_DNN(dnn1, X_tr, epochs=epochs_rbm, lr=lr, batch_size=batch_size)
    print("\n[Training DNN1 (pretrained) with backprop]")
    dnn1, loss1 = retropropagation(dnn1, X_tr, Y_tr, epochs=epochs_bp, lr=lr, batch_size=batch_size)

    print("\n[Training DNN2 (random init) with backprop]")
    dnn2, loss2 = retropropagation(dnn2, X_tr, Y_tr, epochs=epochs_bp, lr=lr, batch_size=batch_size)

    return (
        test_DNN(dnn1, X_tr, Y_tr),
        test_DNN(dnn1, X_te, Y_te),
        test_DNN(dnn2, X_tr, Y_tr),
        test_DNN(dnn2, X_te, Y_te),
        loss1,
        loss2,
    )


# Fig 1: Error vs number of layers (200 neurons/layer)  
print("\n=== Fig 1: layers ===")
n_layers_list = [2, 3, 4, 5]
res_layers = []
for n_l in n_layers_list:
    sizes = [input_dim] + [200] * n_l + [n_classes]
    print(f"\nArchitecture: {sizes}")
    res_layers.append(run_experiment(X_train, Y_train, X_test, Y_test, sizes))

# Save loss curves for each layer config
for idx, n_l in enumerate(n_layers_list):
    loss1 = res_layers[idx][4]
    loss2 = res_layers[idx][5]
    plt.figure()
    plt.plot(range(1, len(loss1)+1), loss1, 'b-', label='Pretrained')
    plt.plot(range(1, len(loss2)+1), loss2, 'r-', label='Random init')
    plt.xlabel('Epoch'); plt.ylabel('Cross-entropy loss')
    plt.title(f'Loss curves — {n_l} hidden layers (200 neurons)')
    plt.legend(); plt.grid(True)
    plt.savefig(f'outputs/fig1_loss_layers{n_l}.png')
    plt.close()

res = torch.tensor([r[:4] for r in res_layers])
plt.figure()
plt.plot(n_layers_list, res[:, 1], 'b-o', label='Pretrained (test)')
plt.plot(n_layers_list, res[:, 3], 'r-o', label='Random (test)')
plt.xlabel('Number of hidden layers')
plt.ylabel('Error rate')
plt.title('Error vs number of layers (200 neurons/layer)')
plt.legend(); plt.grid(True)
plt.savefig('outputs/fig1_layers.png')
print("Saved fig1_layers.png + loss curves")

# Fig 2: Error vs neurons per layer (2 hidden layers)  
print("\n=== Fig 2: neurons ===")
neurons_list = [100, 200, 300, 500, 700]
res_neurons = []
for n_neu in neurons_list:
    sizes = [input_dim, n_neu, n_neu, n_classes]
    print(f"\nArchitecture: {sizes}")
    res_neurons.append(run_experiment(X_train, Y_train, X_test, Y_test, sizes))

# Save loss curves for each neuron config
for idx, n_neu in enumerate(neurons_list):
    loss1 = res_neurons[idx][4]
    loss2 = res_neurons[idx][5]
    plt.figure()
    plt.plot(range(1, len(loss1)+1), loss1, 'b-', label='Pretrained')
    plt.plot(range(1, len(loss2)+1), loss2, 'r-', label='Random init')
    plt.xlabel('Epoch'); plt.ylabel('Cross-entropy loss')
    plt.title(f'Loss curves — 2 layers x {n_neu} neurons')
    plt.legend(); plt.grid(True)
    plt.savefig(f'outputs/fig2_loss_neurons{n_neu}.png')
    plt.close()

res = torch.tensor([r[:4] for r in res_neurons])
plt.figure()
plt.plot(neurons_list, res[:, 1], 'b-o', label='Pretrained (test)')
plt.plot(neurons_list, res[:, 3], 'r-o', label='Random (test)')
plt.xlabel('Neurons per layer')
plt.ylabel('Error rate')
plt.title('Error vs neurons per layer (2 hidden layers)')
plt.legend(); plt.grid(True)
plt.savefig('outputs/fig2_neurons.png')
print("Saved fig2_neurons.png + loss curves")

# Fig 3: Error vs training set size (2 layers x 200)  
print("\n=== Fig 3: data size ===")
n_train_list = [1000, 3000, 7000, 10000, 30000, 60000]
sizes = [input_dim, 200, 200, n_classes]
res_data = []
for n_tr in n_train_list:
    print(f"\nTraining samples: {n_tr}")
    res_data.append(run_experiment(X_train[:n_tr], Y_train[:n_tr], X_test, Y_test, sizes))

# Save loss curves for each data size
for idx, n_tr in enumerate(n_train_list):
    loss1 = res_data[idx][4]
    loss2 = res_data[idx][5]
    plt.figure()
    plt.plot(range(1, len(loss1)+1), loss1, 'b-', label='Pretrained')
    plt.plot(range(1, len(loss2)+1), loss2, 'r-', label='Random init')
    plt.xlabel('Epoch'); plt.ylabel('Cross-entropy loss')
    plt.title(f'Loss curves — {n_tr} training samples (2x200)')
    plt.legend(); plt.grid(True)
    plt.savefig(f'outputs/fig3_loss_ntr{n_tr}.png')
    plt.close()

res = torch.tensor([r[:4] for r in res_data])
plt.figure()
plt.plot(n_train_list, res[:, 1], 'b-o', label='Pretrained (test)')
plt.plot(n_train_list, res[:, 3], 'r-o', label='Random (test)')
plt.xlabel('Number of training samples')
plt.ylabel('Error rate')
plt.title('Error vs training set size (2 layers x 200)')
plt.legend(); plt.grid(True)
plt.savefig('outputs/fig3_data.png')
print("Saved fig3_data.png + loss curves")

print("\nAll done.")
