import numpy as np

def smooth_labels(y_onehot, eps=0.02):
    K = y_onehot.shape[0]

    return y_onehot * (1.0 - eps) + eps / K

y = np.zeros(10)
y[4] = 1

print(smooth_labels(y))