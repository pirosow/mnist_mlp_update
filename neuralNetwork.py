import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, load=False):
        self.w1 = np.random.uniform(-1, 1, size=(hidden_size, input_size))
        self.w2 = np.random.uniform(-1, 1, size=(output_size, hidden_size))

        if load:
            print("loading...")

            try:
                self.w1 = np.load('w1.npy')
                self.w2 = np.load('w2.npy')

            except:
                raise("Could not load network weights.")

        self.b1 = np.random.uniform(-1, 1, size=(hidden_size, 1))
        self.b2 = np.random.uniform(-1, 1, size=(output_size, 1))

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

        self.adam_m1 = np.zeros_like(self.w1)
        self.adam_m2 = np.zeros_like(self.w2)
        self.adam_mb1 = np.zeros_like(self.b1)
        self.adam_mb2 = np.zeros_like(self.b2)

        self.adam_v1 = np.zeros_like(self.w1)
        self.adam_v2 = np.zeros_like(self.w2)
        self.adam_vb1 = np.zeros_like(self.b1)
        self.adam_vb2 = np.zeros_like(self.b2)

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

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

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
        self.x = x

        self.z1 = np.dot(self.w1, x) + self.b1.reshape(1, -1)
        self.a1 = self.relu(self.z1[0])

        self.z2 = np.dot(self.w2, self.a1) + self.b2.reshape(1, -1)
        self.a2 = self.softmax(self.z2[0])

        return self.a2

    def calculate_gradients(self, y_true):
        error_derivative = self.cross_entropy_derivative(self.a2, y_true)  # Use CE derivative
        activation_derivative = 1  # Softmax derivative is handled in CE

        delta2 = error_derivative * activation_derivative

        delta2 = delta2.reshape(len(self.w2), 1)

        self.a1 = self.a1.reshape(len(self.w1), 1)

        gradients2 = np.dot(delta2, self.a1.T)
        gradientb2 = delta2

        delta1 = np.dot(self.w2.T, delta2)

        sum_derivative = self.relu_derivative(self.z1)
        delta1 = delta1 * sum_derivative.reshape(len(self.w1), 1)

        delta1 = delta1.reshape(-1, 1)

        gradients1 = np.dot(delta1, self.x.reshape(1, -1))
        gradientb1 = delta1

        return gradients2, gradients1, gradientb2, gradientb1

    def adam_optimize(self, grads, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads ** 2)

        t = t + 1

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        return m_hat / (np.sqrt(v_hat) + eps), m, v, t

    def augment_weights(self, y_true):
        self.error += self.cross_entropy(self.a2, y_true)

        gradients2, gradients1, gradientb2, gradientb1 = self.calculate_gradients(y_true)

        if self.gradients2 is not None:
            self.gradients2 += gradients2
            self.gradients1 += gradients1
            self.gradientb2 += gradientb2
            self.gradientb1 += gradientb1

        else:
            self.gradients2 = gradients2
            self.gradients1 = gradients1
            self.gradientb2 = gradientb2
            self.gradientb1 = gradientb1

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

        # store optimizer state back
        self.adam_m2, self.adam_v2, self.adam_t2 = m2, v2, t2
        self.adam_m1, self.adam_v1, self.adam_t1 = m1, v1, t1
        self.adam_mb2, self.adam_vb2, self.adam_tb2 = mb2, vb2, tb2
        self.adam_mb1, self.adam_vb1, self.adam_tb1 = mb1, vb1, tb1

        # Standard parameter update (Adam step)
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

        # clear accumulators (same as before)
        self.gradients2 = None
        self.gradients1 = None
        self.gradientb2 = None
        self.gradientb1 = None

        self.error = 0

        return error