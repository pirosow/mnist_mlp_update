import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, load=False):
        if load:
            print("Loading network...")

            try:
                data = np.load("network.npz")

                self.w1 = data["w1"]
                self.w2 = data["w2"]

                self.b1 = data["b1"]
                self.b2 = data["b2"]

            except:
                raise("Could not load network weights.")

        else:
            self.w1 = (np.random.randn(hidden_size, input_size).astype(np.float32) * np.sqrt(2.0 / input_size))
            self.w2 = (np.random.randn(output_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / hidden_size))

            self.b1 = np.zeros((hidden_size)).astype(np.float32)
            self.b2 = np.zeros((output_size)).astype(np.float32)

        self.batch = 0

        self.x = None

        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

        self.gradients2 = None
        self.gradients1 = None
        self.gradientb2 = None
        self.gradientb1 = None

        self.error = 0

        self.adam_m1 = np.zeros_like(self.w1).astype(np.float32)
        self.adam_m2 = np.zeros_like(self.w2).astype(np.float32)
        self.adam_mb1 = np.zeros_like(self.b1).astype(np.float32)
        self.adam_mb2 = np.zeros_like(self.b2).astype(np.float32)

        self.adam_v1 = np.zeros_like(self.w1).astype(np.float32)
        self.adam_v2 = np.zeros_like(self.w2).astype(np.float32)
        self.adam_vb1 = np.zeros_like(self.b1).astype(np.float32)
        self.adam_vb2 = np.zeros_like(self.b2).astype(np.float32)

        self.adam_t1 = 0
        self.adam_t2 = 0
        self.adam_tb1 = 0
        self.adam_tb2 = 0

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, z):
        # z: (B, C)
        z = z - np.max(z, axis=1, keepdims=True)  # numeric stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred + 1e-8))  # Add epsilon to avoid log(0)

    def cross_entropy_derivative(self, y_pred, y_true):
        return (y_pred - y_true)

    def mse(self, y, y_true):
        errors = (y - y_true) ** 2

        error = np.sum(errors) / len(errors)

        return error

    def mse_derivative(self, y, y_true):
        errors = (y - y_true)

        derivative = (2 / len(errors)) * errors

        return np.mean(derivative)

    def forward(self, x):
        x = np.asarray(x)

        if x.ndim == 1:
            x = x.reshape(1, -1)  # (1, D)
        # optional: if user accidentally passes (D, B) transpose:
        if x.ndim == 2 and x.shape[1] != self.w1.shape[1] and x.shape[0] == self.w1.shape[1]:
            x = x.T

        self.x = x

        self.z1 = np.dot(x, self.w1.T) + self.b1.reshape(1, -1)
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.w2.T) + self.b2.reshape(1, -1)
        self.a2 = self.softmax(self.z2)

        return self.a2

    def calculate_gradients(self, y_true):
        """
        Vectorized gradients for a batch.
        Assumes:
          - self.x shape (N, D)
          - self.a1 shape (N, H)
          - self.a2 shape (N, C)
          - self.z1 shape (N, H)
          - self.w2 shape (C, H)
          - y_true either:
              * shape (N,) with class indices, or
              * shape (N, C) one-hot
        Returns:
          grads2 (C,H), grads1 (H,D), gradb2 (C,), gradb1 (H,)
        """

        x = self.x  # (N, D)
        a1 = self.a1  # (N, H)
        a2 = self.a2  # (N, C)
        N = x.shape[0]

        # --- convert labels to one-hot if needed ---
        y = np.asarray(y_true)
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            # integer class labels
            labels = y.reshape(-1)
            y_onehot = np.zeros_like(a2)
            y_onehot[np.arange(N), labels] = 1.0
        else:
            y_onehot = y.reshape(N, -1)  # assume already one-hot (N, C)

        # --- delta on output (softmax + CE) ---
        # use averaged gradient across batch (divide by N)
        delta2 = (a2 - y_onehot) / N  # (N, C)

        # grads for w2 and b2
        # w2 has shape (C, H) so gradient is delta2^T @ a1 -> (C, H)
        gradients2 = delta2.T.dot(a1)  # (C, H)
        gradientb2 = np.sum(delta2, axis=0)  # (C,)

        # backprop into hidden layer
        # delta1 shape (N, H): delta2 @ w2 -> (N, H)
        delta1 = delta2.dot(self.w2)  # (N, H)
        # multiply by ReLU derivative (shape (N,H))
        delta1 = delta1 * self.relu_derivative(self.z1)

        # grads for w1 and b1
        gradients1 = delta1.T.dot(x)  # (H, D)
        gradientb1 = np.sum(delta1, axis=0)  # (H,)

        # keep dtype consistent with optimizer
        return (gradients2.astype(np.float32),
                gradients1.astype(np.float32),
                gradientb2.astype(np.float32),
                gradientb1.astype(np.float32))

    def adam_optimize(self, grads, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads ** 2)

        t = t + 1

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        return m_hat / (np.sqrt(v_hat) + eps), m, v, t

    def get_loss(self, a, y_true):
        return self.cross_entropy(a, y_true)

    def augment_weights(self, y_true):
        self.error += self.cross_entropy(self.a2, y_true)

        gradients2, gradients1, gradientb2, gradientb1 = self.calculate_gradients(y_true)

        if self.gradients2 is not None:
            self.gradients2 += gradients2
            self.gradients1 += gradients1
            self.gradientb2 += gradientb2
            self.gradientb1 += gradientb1

        else:
            self.gradients2 = gradients2.astype(np.float32)
            self.gradients1 = gradients1.astype(np.float32)
            self.gradientb2 = gradientb2.astype(np.float32)
            self.gradientb1 = gradientb1.astype(np.float32)

    def update_weights(self, batches, lr=0.01, weight_decay=1e-4):
        """
        Update weights using your adam_optimize helper and apply decoupled L2 (AdamW-style).
        - batches: number of minibatches accumulated (you already divide gradients by batches)
        - lr: learning rate
        - weight_decay: lambda; small value e.g. 1e-4
        """
        error = self.error / batches

        # Call your adam optimizer to get preconditioned gradients (same as before)
        gradients2, m2, v2, t2 = self.adam_optimize(self.gradients2 / batches, self.adam_m2, self.adam_v2, self.adam_t2)
        gradients1, m1, v1, t1 = self.adam_optimize(self.gradients1 / batches, self.adam_m1, self.adam_v1, self.adam_t1)
        gradientb2, mb2, vb2, tb2 = self.adam_optimize(self.gradientb2 / batches, self.adam_mb2, self.adam_vb2,
                                                       self.adam_tb2)
        gradientb1, mb1, vb1, tb1 = self.adam_optimize(self.gradientb1 / batches, self.adam_mb1, self.adam_vb1,
                                                       self.adam_tb1)

        self.adam_m2, self.adam_v2, self.adam_t2 = m2, v2, t2
        self.adam_m1, self.adam_v1, self.adam_t1 = m1, v1, t1
        self.adam_mb2, self.adam_vb2, self.adam_tb2 = mb2, vb2, tb2
        self.adam_mb1, self.adam_vb1, self.adam_tb1 = mb1, vb1, tb1

        self.w2 -= lr * gradients2
        self.w1 -= lr * gradients1
        self.b2 -= lr * gradientb2
        self.b1 -= lr * gradientb1

        # --- Decoupled weight decay (AdamW) ---
        # Apply weight decay directly to weights (do NOT decay biases typically)
        if weight_decay and weight_decay > 0.0:
            # multiplicative factor (approx 1 - lr * lambda)
            decay_factor = 1.0 - lr * weight_decay
            # clamp factor to non-negative (in case lr*weight_decay > 1)
            decay_factor = max(decay_factor, 0.0)
            self.w2 *= decay_factor
            self.w1 *= decay_factor
            # NOTE: biases b1,b2 usually not decayed

        self.gradients2 = None
        self.gradients1 = None
        self.gradientb2 = None
        self.gradientb1 = None

        self.error = 0

        return error