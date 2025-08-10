import numpy as np
from copy import deepcopy

class NeuralNetwork:
    def __init__(self, *weights_size, load=False, activation="relu"):
        self.weights = []
        self.biases = []

        for i in range(len(weights_size) - 1):
            w_size = (weights_size[i + 1], weights_size[i])
            b_size = (weights_size[i + 1], 1)

            weights = np.random.uniform(-1, 1, size=w_size).astype(np.float32)
            biases = np.zeros(b_size).astype(np.float32)

            self.weights.append(weights)
            self.biases.append(biases)

        if load:
            print("loading...")
            try:
                # safer load for lists/objects
                self.weights = np.load('weights.npy', allow_pickle=True).tolist()
                # if you saved biases separately, you can load them similarly
            except Exception as e:
                raise RuntimeError("Could not load network weights.") from e

        self.batch = 0

        # caches
        self.z = []   # pre-activations (each (units,1))
        self.a = []   # activations; a[0] will be input column

        self.x = []   # kept for compatibility with older code (we'll keep it in sync)

        self.error = 0.0

        # choose activation
        if activation == "relu":
            self.activation = self.relu
            self.activationD = self.relu_derivative
        elif activation == "sigmoid":
            self.activation = self.sigmoid
            self.activationD = self.sigmoid_derivative
        else:
            raise ValueError("Unsupported activation")

        self.outputActivation = self.softmax
        self.loss = self.cross_entropy
        self.lossD = self.cross_entropy_derivative

        # initialize gradient accumulators (same ordering as weights)
        self.w_gradients = [np.zeros_like(w) for w in self.weights]
        self.b_gradients = [np.zeros_like(b) for b in self.biases]

        # Adam buffers (same ordering as weights)
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.t_w = [0 for _ in self.weights]

        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t_b = [0 for _ in self.biases]

    # ---- activations ----
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0).astype(np.float32)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def softmax(self, x):
        # stable softmax; x expected shape (classes,1)
        ex = np.exp(x - np.max(x, axis=0, keepdims=True))
        return ex / ex.sum(axis=0, keepdims=True)

    # ---- losses ----
    def cross_entropy(self, y_pred, y_true):
        # expects column vectors
        return -np.sum(y_true * np.log(y_pred + 1e-8))

    def cross_entropy_derivative(self, y_pred, y_true):
        return (y_pred - y_true)  # for single-sample column vectors

    def mse(self, y, y_true):
        errors = (y - y_true) ** 2
        error = np.sum(errors) / len(errors)
        return error

    def mse_derivative(self, y, y_true):
        errors = (y - y_true)
        derivative = (2 / len(errors)) * errors
        return np.mean(derivative)

    # ---- forward (single sample) ----
    def forward(self, x):
        # ensure column vector
        x = np.asarray(x).reshape(-1, 1)
        self.x = [x]      # keep compatibility if other parts read self.x
        self.z = []
        self.a = [x]      # a[0] is input column

        # all hidden layers except last
        for i in range(len(self.weights) - 1):
            w = self.weights[i]    # (out, in)
            b = self.biases[i]     # (out, 1)
            z = np.dot(w, self.a[-1]) + b      # (out,1)
            a = self.activation(z)             # (out,1)
            self.z.append(z)
            self.a.append(a)
            self.x.append(a)  # keep x in sync

        # final layer
        w = self.weights[-1]
        b = self.biases[-1]
        z = np.dot(w, self.a[-1]) + b
        a = self.outputActivation(z)
        self.z.append(z)
        self.a.append(a)
        self.x.append(a)

        # return flattened array so code using np.argmax(nn.forward(x)) still works
        return a.ravel()

    # ---- backprop for single sample ----
    def calculate_gradients2(self, y_true):
        # ensure y is column vector
        y = np.asarray(y_true).reshape(-1, 1)

        L = len(self.weights)
        w_gradients = [None] * L
        b_gradients = [None] * L

        # last-layer delta
        delta = self.lossD(self.a[-1], y)  # (out,1)

        # gradient for last layer
        a_prev = self.a[-2]  # (in,1)
        w_gradients[L - 1] = np.dot(delta, a_prev.T)  # (out, in)
        b_gradients[L - 1] = delta.copy()             # (out,1)

        last_delta = delta

        # propagate backward
        for l in range(L - 2, -1, -1):
            W_next = self.weights[l + 1]   # (out_next, out_curr)
            Z_curr = self.z[l]             # (out_curr,1)
            delta = np.dot(W_next.T, last_delta) * self.activationD(Z_curr)  # (out_curr,1)
            a_prev = self.a[l]             # (in,1)
            w_gradients[l] = np.dot(delta, a_prev.T)
            b_gradients[l] = delta.copy()
            last_delta = delta

        return w_gradients, b_gradients

    # ---- Adam helper (kept as your pattern) ----
    def adam_optimize(self, grads, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads ** 2)
        t = t + 1
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        return m_hat / (np.sqrt(v_hat) + eps), m, v, t

    # ---- accumulate gradients ----
    def augment_weights(self, y_true):
        self.error += self.loss(self.a[-1], y_true)
        wgs, bgs = self.calculate_gradients2(y_true)

        # accumulate; add defensive checks
        for i in range(len(self.weights)):
            if self.w_gradients[i].shape != wgs[i].shape:
                raise ValueError(f"w_grad shape mismatch layer {i}: {self.w_gradients[i].shape} vs {wgs[i].shape}")
            if self.b_gradients[i].shape != bgs[i].shape:
                raise ValueError(f"b_grad shape mismatch layer {i}: {self.b_gradients[i].shape} vs {bgs[i].shape}")
            self.w_gradients[i] += wgs[i]
            self.b_gradients[i] += bgs[i]

    def update_weights(self, batches, lr, weight_decay=1e-4):
        """
        Apply accumulated gradients using Adam (per-parameter) and decoupled weight decay.
        - batches: number of accumulated samples (we divide accumulators by batches)
        - lr: learning rate
        - weight_decay: decoupled L2 (AdamW style). Biases are not decayed.
        """
        if batches <= 0:
            raise ValueError("batches must be > 0")

        error = self.error / batches

        # For each layer, compute averaged grads then run Adam update for weights and biases
        for i in range(len(self.weights)):
            grad_w = self.w_gradients[i] / batches  # same shape as self.weights[i]
            grad_b = self.b_gradients[i] / batches  # same shape as self.biases[i]

            # Adam for weights
            step_w, new_mw, new_vw, new_tw = self.adam_optimize(grad_w, self.m_w[i], self.v_w[i], self.t_w[i])
            self.m_w[i], self.v_w[i], self.t_w[i] = new_mw, new_vw, new_tw
            self.weights[i] -= lr * step_w

            # Adam for biases
            step_b, new_mb, new_vb, new_tb = self.adam_optimize(grad_b, self.m_b[i], self.v_b[i], self.t_b[i])
            self.m_b[i], self.v_b[i], self.t_b[i] = new_mb, new_vb, new_tb
            self.biases[i] -= lr * step_b

            # decoupled weight decay (do NOT decay biases)
            if weight_decay and weight_decay > 0.0:
                decay_factor = max(1.0 - lr * weight_decay, 0.0)
                self.weights[i] *= decay_factor

        # reset accumulators
        self.w_gradients = [np.zeros_like(w) for w in self.weights]
        self.b_gradients = [np.zeros_like(b) for b in self.biases]

        self.error = 0.0
        return error