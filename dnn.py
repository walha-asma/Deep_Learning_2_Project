import torch
import copy
from rbm import entree_sortie_RBM, device
from dbn import init_DBN, train_DBN


def init_DNN(layer_sizes):
    return init_DBN(layer_sizes)


def pretrain_DNN(dnn, X, epochs=100, lr=0.1, batch_size=128):
    dbn_part = train_DBN(dnn[:-1], X, epochs, lr, batch_size)
    dnn[:-1] = dbn_part
    return dnn


def calcul_softmax(rbm, X):
    logits = X @ rbm['W'] + rbm['b']
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp = torch.exp(logits)
    return exp / exp.sum(dim=1, keepdim=True)


def entree_sortie_reseau(dnn, X):
    outputs = [X]
    current = X
    for rbm in dnn[:-1]:
        current = entree_sortie_RBM(rbm, current)
        outputs.append(current)
    probs = calcul_softmax(dnn[-1], current)
    outputs.append(probs)
    return outputs


def retropropagation(dnn, X, Y, epochs=200, lr=0.1, batch_size=128):
    X, Y = X.to(device), Y.to(device)
    n = X.shape[0]
    loss_history = []

    for epoch in range(epochs):
        indices = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            bx, by = X[idx], Y[idx]
            m = bx.shape[0]

            outputs = entree_sortie_reseau(dnn, bx)

            # Output layer delta (softmax + cross-entropy)
            delta = outputs[-1] - by

            for i in reversed(range(len(dnn))):
                h_prev = outputs[i]
                dW = h_prev.T @ delta / m
                db = delta.mean(dim=0)

                if i > 0:
                    delta = (delta @ dnn[i]['W'].T) * outputs[i] * (1 - outputs[i])

                dnn[i]['W'] -= lr * dW
                dnn[i]['b'] -= lr * db

        # Cross-entropy loss after parameter update
        with torch.no_grad():
            probs = torch.clamp(entree_sortie_reseau(dnn, X)[-1], 1e-10, 1)
            ce = -(Y * torch.log(probs)).sum(dim=1).mean().item()
        loss_history.append(ce)
        print(f"Epoch {epoch+1}/{epochs} - Cross-entropy: {ce:.4f}")
    return dnn, loss_history


def test_DNN(dnn, X, Y):
    X, Y = X.to(device), Y.to(device)
    with torch.no_grad():
        probs = entree_sortie_reseau(dnn, X)[-1]
    preds = probs.argmax(dim=1)
    true = Y.argmax(dim=1)
    return (preds != true).float().mean().item()
