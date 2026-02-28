import numpy as np
from rbm import init_RBM, entree_sortie_RBM, sigmoid
from dbn import init_DBN, train_DBN


def init_DNN(layer_sizes):
    # layer_sizes includes input, hidden layers, and output (n_classes)
    return init_DBN(layer_sizes)


def pretrain_DNN(dnn, X, epochs=100, lr=0.1, batch_size=128):
    # Only pretrain hidden layers (all but last which is classification layer)
    dbn_part = dnn[:-1]
    dbn_part = train_DBN(dbn_part, X, epochs, lr, batch_size)
    dnn[:-1] = dbn_part
    return dnn


def calcul_softmax(rbm, X):
    logits = X @ rbm['W'] + rbm['b']
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


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
    n = X.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            batch_X, batch_Y = X[idx], Y[idx]
            m = batch_X.shape[0]
            
            # Forward pass
            outputs = entree_sortie_reseau(dnn, batch_X)
            
            # Backprop - output layer (softmax + cross-entropy)
            delta = outputs[-1] - batch_Y
            
            for i in reversed(range(len(dnn))):
                h_prev = outputs[i]
                
                dW = h_prev.T @ delta / m
                db = np.mean(delta, axis=0)
                
                if i > 0:
                    da = np.mean(delta @ dnn[i]['W'].T, axis=0) if i == len(dnn)-1 else np.mean(delta, axis=0)
                    # Backprop through sigmoid
                    delta_prev = (delta @ dnn[i]['W'].T) * outputs[i] * (1 - outputs[i])
                else:
                    delta_prev = None
                
                dnn[i]['W'] -= lr * dW
                dnn[i]['b'] -= lr * db
                
                if delta_prev is not None:
                    delta = delta_prev
        
        # Compute cross-entropy loss
        all_out = entree_sortie_reseau(dnn, X)
        probs = np.clip(all_out[-1], 1e-10, 1)
        ce = -np.mean(np.sum(Y * np.log(probs), axis=1))
        print(f"Epoch {epoch+1}/{epochs} - Cross-entropy: {ce:.4f}")
    return dnn


def test_DNN(dnn, X_test, Y_test):
    outputs = entree_sortie_reseau(dnn, X_test)
    preds = np.argmax(outputs[-1], axis=1)
    true = np.argmax(Y_test, axis=1)
    error_rate = np.mean(preds != true)
    return error_rate
