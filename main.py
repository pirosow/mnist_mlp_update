import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from neuralNetwork import NeuralNetwork
import numpy as np
from tensorflow.keras.datasets import mnist
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from threading import Thread
import random
from PIL import Image
import time

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

epochs = 10

min_lr = 0.0001
max_lr = 0.01

# Initialize neural network and data lists
nn = NeuralNetwork(784, 128, 10, load=False)
errors = []
gens = []
accuracies = []
accuracy_gens = []

n_error = 1000
n_accuracy = 10000

gen = 0
epoch = 0

# Create Dash app
app = dash.Dash(__name__)

# Layout with graph and hidden interval component
app.layout = html.Div([
    dcc.Graph(id='live-graph'),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])

# Initialize figure with two y-axes
fig = go.Figure()
fig.add_trace(go.Scatter(x=gens, y=errors, mode='lines', name='Error', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=accuracy_gens, y=accuracies, mode='lines', name='Accuracy',
                         line=dict(color='red'), yaxis='y2'))

fig.update_layout(
    title='Error and Accuracy vs Generation (Training Progress)',
    xaxis_title='Generation (k)',
    yaxis=dict(title='Error', color='blue'),
    yaxis2=dict(title='Accuracy (%)', color='red', overlaying='y', side='right'),
)

def augment_image(img_array, noise_multiplier, rotation_threshold):
    """Robust MNIST augmentation with dimension checks"""
    # Convert to 2D uint8 for processing
    if img_array.ndim == 3:
        processed = (img_array.squeeze() * 255).astype(np.uint8)
    else:
        processed = (img_array * 255).astype(np.uint8)

    # Rotate with black background
    img = Image.fromarray(processed, mode='L')

    rotation = random.randint(-rotation_threshold, rotation_threshold)

    rotated = img.rotate(rotation, fillcolor=0)

    # Convert back to array
    rotated_array = np.array(rotated)

    # Add noise and clip
    noise = np.random.normal(0, noise_multiplier, rotated_array.shape)
    noisy_array = rotated_array.astype(np.float32) + noise
    noisy_array = np.clip(noisy_array, 0, 255)

    # Return normalized with channel dimension
    return (noisy_array / 255.0).reshape(28, 28, 1)


@app.callback(
    Output('live-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    global fig
    fig.data[0].x = gens
    fig.data[0].y = errors
    fig.data[1].x = accuracy_gens
    fig.data[1].y = accuracies
    return fig


def test_accuracy(samples=1000):
    good = 0

    for _ in range(samples):
        j = random.randint(0, len(x_test) - 1)
        x = x_test[j].flatten()

        #x = augment_image(x, 30, 50).flatten()

        y = y_test[j]
        prediction = np.argmax(nn.forward(x))
        good += int(prediction == y)

    return round((good / samples) * 10000) / 100

def training_loop():
    global gen, epoch, errors, gens, accuracies, accuracy_gens

    step = (max_lr - min_lr) / epochs

    lr = max_lr + step

    time.sleep(0.1)

    start_time = time.time()

    print("\nStarted training \n")

    print(f"Total epochs: {epochs}")

    print(f"Max lr: {max_lr} \nMin lr: {min_lr} \n")

    for epoch in range(1, epochs + 1):
        lr -= step

        print(f"Epoch {epoch}/{epochs}")
        print(f"Lr: {lr} \n")

        # Test accuracy
        accuracy = test_accuracy(1000)
        accuracies.append(accuracy)
        accuracy_gens.append(gen // n_error)

        print(f"Accuracy: {accuracy}% \n")

        print(f"Train time: {round(time.time() - start_time)} seconds")

        # Training iterations
        for i in range(len(x_train)):
            gen += 1

            # In the training loop:
            original = x_train[i]

            augmented = augment_image(original, 30, 50)

            # Convert for saving
            save_array = augmented.squeeze()  # Remove channel dimension (28,28,1) -> (28,28)
            save_array = (save_array * 255).astype(np.uint8)  # Convert to 0-255

            # Ensure array is 2D and create proper PIL Image
            if save_array.ndim == 2:
                pil_img = Image.fromarray(save_array, mode='L')  # Explicit grayscale mode
            else:  # Handle rare cases with unexpected dimensions
                pil_img = Image.fromarray(save_array[:, :, 0], mode='L')

            # Continue training
            x = augmented.flatten()

            # Training step
            y = y_train[i]
            y_list = [0] * 10
            y_list[y] = 1

            nn.forward(x)
            nn.update_weights(y_list, lr=lr)

            # Update metrics
            if gen % n_error == 0:
                errors.append(nn.error)
                gens.append(gen // n_error)

            if gen % n_accuracy == 0:
                accuracy = test_accuracy(1000)
                accuracies.append(accuracy)
                accuracy_gens.append(gen // n_error)

        # Save weights
        np.savetxt('w1.npy', nn.w1)
        np.savetxt('w2.npy', nn.w2)

        print("")

    quit(0)

# Start training thread
Thread(target=training_loop, daemon=True).start()

# Run server
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)