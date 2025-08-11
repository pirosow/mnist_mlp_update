import os
from copy import deepcopy

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from neuralNetworkMultipleLayers import NeuralNetwork
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

epochs = 100000

min_lr = 0.000001 #0.01
max_lr = 0.001 #0.25

load = False

# Initialize neural network and data lists
nn = NeuralNetwork(784, 256, 128, 10, load=load)
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

def updateStats():
    global avg_error, accuraciesTest, accuraciesTrain, accuracy_gens, errors, lr, start_time, pil_img

    accuracyTest, accuracyTrain = test_accuracy(1000)
    accuraciesTest.append(accuracyTest)
    accuraciesTrain.append(accuracyTrain)
    accuracy_gens.append(gen // n_error)

    errors.append(avg_error)

    # Save weights
    #np.save('weights.npy', nn.weights)

    print(f"Test accuracy: {accuracyTest}%")
    print(f"Train accuracy: {accuracyTrain}%")

def training_loop():
    global gen, epoch, errors, gens, accuracies, accuracy_gens, accuracyTest, accuracyTrain, updateStep, avg_error, pil_img

    batches = 64

    step = (max_lr - min_lr) / epochs

    lr = max_lr + step

    time.sleep(0.1)

    start_time = time.time()

    print("\nStarted training \n")

    print(f"Total epochs: {epochs}")

    print(f"Max lr: {max_lr} \nMin lr: {min_lr} \n")

    time.sleep(2)

    for epoch in range(1, epochs + 1):
        lr -= step

        gen += 1

        indices = np.arange(len(x_train))
        np.random.shuffle(indices)

        # Training iterations
        for i in range(batches):
            # In the training loop:
            index = indices[i]

            original = x_train[index]

            augmented = augment_image(original, image_noise, rotation, move)

            # Continue training
            x = augmented.flatten()

            # Training step
            y = y_train[index]
            y_list = np.zeros((10, 1))
            y_list[y] = 1

            y_list = smooth_labels(y_list)

            nn.forward(x)

            nn.augment_weights(y_list)

        avg_error = nn.update_weights(batches, lr=lr)

        if epoch % updateStep == 0:
            print("\n\n\n\n\n")

            print(f"Epoch {epoch}/{epochs}")
            print(f"Lr: {lr} \n")
            print(f"Train time: {round(time.time() - start_time)} seconds")

            updateStats()

            # Convert for saving
            save_array = augmented.squeeze()  # Remove channel dimension (28,28,1) -> (28,28)
            save_array = (save_array * 255).astype(np.uint8)  # Convert to 0-255

            # Ensure array is 2D and create proper PIL Image
            if save_array.ndim == 2:
                pil_img = Image.fromarray(save_array, mode='L')  # Explicit grayscale mode
            else:  # Handle rare cases with unexpected dimensions
                pil_img = Image.fromarray(save_array[:, :, 0], mode='L')

            pil_img.save("training_example.png")

    quit(0)

training_loop()