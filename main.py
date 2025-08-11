import os
from copy import deepcopy

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
import threading
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

x_test_flat = x_test.reshape(len(x_test), -1)
x_train_flat = x_train.reshape(len(x_train), -1)

image_noise = 20
rotation = 50
move = 4

epochs = 10
batch_size = 64

min_lr = 0.00001 #0.01
max_lr = 0.001 #0.25

load = bool(input("Do you want to load the last training (y) or train from scratch (n)? (y/n)").lower() in ["y", "yes"])

# Initialize neural network and data lists
nn = NeuralNetwork(784, 1024, 10, load=load)
errors = []
gens = []
accuraciesTest = []
accuraciesTrain = []
accuracy_gens = []

n_error = 1
updateStep = 25

accuracyTest = 0
accuracyTrain = 0

gen = 0
epoch = 0

# Create Dash app
app = dash.Dash(__name__)

# Layout with graph and hidden interval component
app.layout = html.Div([
    dcc.Graph(id='live-graph'),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])

# Utility: simple rolling average function
def rolling_avg(lst, window=5):
    out = []
    for i in range(len(lst)):
        window_vals = lst[max(0, i - window + 1): i + 1]
        if len(window_vals) == 0:
            out.append(None)
        else:
            out.append(float(np.mean(window_vals)))
    return out

# Initialize figure with two axes only: errors on y (primary), accuracies on y2 (secondary)
fig = go.Figure()

# Trace 0: raw error (primary y)
fig.add_trace(go.Scatter(x=accuracy_gens, y=errors, mode='lines', name='Error', line=dict(color='blue'), yaxis='y'))

# Trace 1: avg error (last 5) on the same error axis
fig.add_trace(go.Scatter(x=accuracy_gens, y=[], mode='lines', name='Avg error (last 5)', line=dict(color='purple', width=4), yaxis='y'))

# Trace 2: test accuracy (rolling avg last 5) on accuracy axis y2
fig.add_trace(go.Scatter(x=accuracy_gens, y=[], mode='lines', name='Test accuracy (avg last 5)', line=dict(color='green'), yaxis='y2'))

# Trace 3: train accuracy (rolling avg last 5) on accuracy axis y2
fig.add_trace(go.Scatter(x=accuracy_gens, y=[], mode='lines', name='Train accuracy (avg last 5)', line=dict(color='yellow'), yaxis='y2'))

fig.update_layout(
    title='Error and Accuracy vs Generation (Training Progress)',
    xaxis_title='Generation (k)',
    # Primary error axis (put on the right)
    yaxis=dict(
        title='Error',
        color='blue',
        side='left',
    ),
    # Single accuracy axis for both accuracy traces (also on the right, slightly further)
    yaxis2=dict(
        title='Accuracy (%) (avg last 5)',
        color='green',
        overlaying='y',
        side='right',
    ),
    margin=dict(l=60, r=160, t=60, b=60),
    legend=dict(y=0.99, x=0.01),
    hovermode="x unified"
)

from PIL import Image

def move_image(image, x, y, fill=0):
    """
    Move the image by x pixels on both the x and y axes.

    Parameters:
        image (PIL.Image.Image): The input image.
        x (int): Number of pixels to shift along both axes.
        fill (int/tuple): Fill color for exposed areas (default is 0, black).

    Returns:
        PIL.Image.Image: The shifted image.
    """
    return image.transform(
        image.size,
        Image.AFFINE,
        (1, 0, -x, 0, 1, -y),
        fillcolor=fill
    )

def smooth_labels(y_onehot, eps=0.02):
    K = y_onehot.shape[0]

    return y_onehot * (1.0 - eps) + eps / K

def onehot_batch(mini_set):
    batch = np.zeros((10, len(mini_set)), dtype=np.float32)

    for j, idx in enumerate(mini_set):
        batch[y_train[idx], j] = 1.0

    return smooth_labels(batch)   # implement smoothing to work on arrays


def augment_image(img_array, noise_multiplier, rotation_threshold, move_threshold, scale_threshold=0.30):
    """
    Robust MNIST augmentation with optional scaling (zoom).

    Parameters
    ----------
    img_array : np.ndarray
        Input image, either shape (28,28) or (28,28,1), values in [0,1] or [0,255].
    noise_multiplier : float or int
        Max stddev for additive Gaussian noise (in pixel units 0..255). Per-sample sigma is drawn
        from [noise_multiplier/3, noise_multiplier].
    rotation_threshold : int
        Max absolute degrees for random rotation. Rotation is sampled uniformly in [-rotation_threshold, rotation_threshold].
    move_threshold : int
        Max pixel translation in both x and y; translation is sampled uniformly in [-move_threshold, move_threshold].
    scale_threshold : float (default 0.12)
        Maximum relative scaling. Scale is sampled uniformly from [max(0.5, 1-scale_threshold), 1+scale_threshold].
        For example scale_threshold=0.12 -> scale in [0.88, 1.12] (clamped at a minimum of 0.5).
    """
    # Normalize to uint8 28x28 greyscale
    if img_array.ndim == 3:
        processed = img_array.squeeze()
    else:
        processed = img_array

    # Accept [0,1] floats or [0,255] ints
    if processed.dtype == np.float32 or processed.dtype == np.float64:
        processed = (processed * 255.0).astype(np.uint8)
    else:
        processed = processed.astype(np.uint8)

    # Make PIL image
    img = Image.fromarray(processed, mode='L')

    # --- Scale (zoom) ---
    if scale_threshold is not None and scale_threshold > 0.0:
        low = max(0.5, 1.0 - scale_threshold)
        high = 1.0 + scale_threshold
        scale = random.uniform(low, high)
    else:
        scale = 1.0

    if abs(scale - 1.0) > 1e-6:
        new_size = max(1, int(round(28 * scale)))
        # resize with bilinear (reasonable for small images)
        scaled = img.resize((new_size, new_size), resample=Image.BILINEAR)
        # paste onto black 28x28 canvas, centered
        canvas = Image.new('L', (28, 28), 0)
        paste_x = (28 - new_size) // 2
        paste_y = (28 - new_size) // 2
        canvas.paste(scaled, (paste_x, paste_y))
        img = canvas

    # --- Rotation ---
    if rotation_threshold and rotation_threshold > 0:
        rotation = random.randint(-rotation_threshold, rotation_threshold)
        img = img.rotate(rotation, fillcolor=0)

    # --- Translation / move ---
    if move_threshold and move_threshold > 0:
        dx = random.randint(-move_threshold, move_threshold)
        dy = random.randint(-move_threshold, move_threshold)
    else:
        dx = dy = 0
    moved = Image.new('L', (28, 28), 0)
    # paste accepts negative coords -> crops appropriately, leaving background black
    moved.paste(img, (dx, dy))
    img = moved

    # Convert back to numpy (uint8)
    arr = np.array(img).astype(np.float32)

    # --- Noise: per-sample sigma ---
    # allow noise_multiplier to be float; fallback to small value if <=0
    max_sigma = float(noise_multiplier) if noise_multiplier is not None else 0.0
    if max_sigma <= 0:
        sigma = 0.0
    else:
        min_sigma = max_sigma / 3.0
        sigma = random.uniform(min_sigma, max_sigma)

    if sigma > 0:
        noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape)
        arr = arr + noise

    arr = np.clip(arr, 0, 255)

    # Return normalized to [0,1] with channel dim
    out = (arr / 255.0).astype(np.float32).reshape(28, 28, 1)

    return out


@app.callback(
    Output('live-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    global fig

    # convert to plain lists to avoid JSON serialization problems
    x_vals = list(accuracy_gens)

    # update raw error line (trace 0)
    fig.data[0].x = x_vals
    fig.data[0].y = list(errors)

    # compute rolling average over last 5 errors and set trace 1
    err_avg5 = rolling_avg(list(errors), window=5)
    fig.data[1].x = x_vals
    fig.data[1].y = err_avg5

    # For accuracies, plot rolling average (last 5)
    test_avg5 = rolling_avg(list(accuraciesTest), window=5)
    train_avg5 = rolling_avg(list(accuraciesTrain), window=5)

    fig.data[2].x = x_vals
    fig.data[2].y = test_avg5

    fig.data[3].x = x_vals
    fig.data[3].y = train_avg5

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

    accTest = round((good / samples) * 10000) / 100

    good = 0

    for _ in range(samples):
        j = random.randint(0, len(x_train) - 1)
        original = x_train[j]

        augmented = augment_image(original, image_noise, rotation, move)

        # Continue training
        x = augmented.flatten()

        y = y_train[j]
        prediction = np.argmax(nn.forward(x))
        good += int(prediction == y)

    accTrain = round((good / samples) * 10000) / 100

    return accTest, accTrain

def full_test():
    rightTest = 0
    rightTrain = 0

    for i, img in enumerate(x_test):
        x = img.flatten()
        y = y_test[i]

        pred = np.argmax(nn.forward(x))

        if pred == y:
            rightTest += 1

    for i, img in enumerate(x_train):
        x = img.flatten()
        y = y_train[i]

        pred = np.argmax(nn.forward(x))

        if pred == y:
            rightTrain += 1

    testAcc = round(rightTest / len(x_test), 4) * 100
    trainAcc = round(rightTrain / len(x_train), 4) * 100

    return testAcc, trainAcc

def augmented(mini_set):
    with ThreadPoolExecutor(max_workers=8) as ex:
        imgs = list(ex.map(lambda i: augment_image(x_train[i], image_noise, rotation, move), mini_set))
    # stack into (784, batch)
    x = np.stack([img.flatten() for img in imgs], axis=1).astype(np.float32)

    return x

def updateStats(gen):
    global avg_error, accuraciesTest, accuraciesTrain, accuracy_gens, errors, batch_size

    accuracyTest, accuracyTrain = test_accuracy(1000)
    accuraciesTest.append(accuracyTest)
    accuraciesTrain.append(accuracyTrain)
    accuracy_gens.append(gen)

    errors.append(avg_error)

    # Save weights
    np.savez('network.npz', w1=nn.w1, w2=nn.w2, b1=nn.b1, b2=nn.b2)

    print(f"Test accuracy: {accuracyTest}%")
    print(f"Train accuracy: {accuracyTrain}%")

def update(generation, epoch, epochs, start_time, x, lr, batch, gen):
    print("\n\n\n\n\n")

    print(f"Epoch {epoch}/{epochs}")
    print(f"Generation {generation}")
    print(f"Image {batch}/{len(x_train)}")
    print(f"Lr: {lr} \n")
    print(f"Train time: {round(time.time() - start_time)} seconds")

    updateStats(gen)

    # Convert for saving
    save_array = x.reshape(28, 28)  # Remove channel dimension (28,28,1) -> (28,28)
    save_array = (save_array * 255).astype(np.uint8)  # Convert to 0-255

    # Ensure array is 2D and create proper PIL Image
    if save_array.ndim == 2:
        pil_img = Image.fromarray(save_array, mode='L')  # Explicit grayscale mode
    else:  # Handle rare cases with unexpected dimensions
        pil_img = Image.fromarray(save_array[:, :, 0], mode='L')

    pil_img.save("training_example.png")

def training_loop():
    global gen, epoch, errors, gens, updateStep, avg_error

    step = (max_lr - min_lr) / epochs

    lr = max_lr + step

    time.sleep(0.1)

    start_time = time.time()

    print("\nStarted training \n")

    print(f"Total epochs: {epochs}")

    print(f"Max lr: {max_lr} \nMin lr: {min_lr} \n")

    time.sleep(2)

    indices = np.arange(len(x_train))

    generation = 0

    for epoch in range(1, epochs + 1):
        lr -= step

        np.random.shuffle(indices)

        for batch in range(0, len(x_train), batch_size):
            generation += 1

            mini_set = indices[batch:batch + batch_size]

            X = augmented(mini_set).T
            Y = onehot_batch(mini_set).T

            # Training iterations
            for j in range(len(mini_set)):
                x = X[j]

                y = Y[j]

                nn.forward(x)

                nn.augment_weights(y)

            avg_error = nn.update_weights(len(mini_set), lr=lr)

            if generation % updateStep == 0:
                Thread(update(generation, epoch, epochs, start_time, x, lr, batch, generation), daemon=True).start()

    quit(0)

# Start training thread
Thread(target=training_loop, daemon=True).start()

# Run server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
