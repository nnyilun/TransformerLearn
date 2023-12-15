import torch

def accuracy(y_hat:torch.Tensor, y:torch.Tensor) -> float:
    y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.type(y.dtype) == y).sum().item()
    return float(cmp / y.size(0))

def get_k_fold_data(k, i, X, y):
    assert k > 1
    
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid