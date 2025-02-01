import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, load=False):
        self.w1 = np.random.uniform(-1, 1, size=(hidden_size, input_size))
        self.w2 = np.random.uniform(-1, 1, size=(output_size, hidden_size))

        if load:
            print("loading...")

            try:
                self.w1 = np.loadtxt('w1.npy')
                self.w2 = np.loadtxt('w2.npy')

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

        self.avg_gradients2 = None
        self.avg_gradients1 = None
        self.avg_gradientb2 = None
        self.avg_gradientb1 = None

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
        return (y_pred - y_true) / y_pred.shape[0]  # Simplified derivative for softmax + CE

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

    def update_weights(self, y_true, lr=0.01):
        self.error = self.cross_entropy(self.a2, y_true)

        gradients2, gradients1, gradientb2, gradientb1 = self.calculate_gradients(y_true)

        self.w2 -= lr * gradients2
        self.w1 -= lr * gradients1
        self.b2 -= lr * gradientb2
        self.b1 -= lr * gradientb1