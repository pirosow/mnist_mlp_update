from neuralNetwork import NeuralNetwork
import numpy as np
import random
from PIL import Image
import time

# Load MNIST dataset
mnist = np.load("mnist/mnist.npz")

x_train, y_train = mnist['x_train'], mnist['y_train']
x_test,  y_test  = mnist['x_test'],  mnist['y_test']

# Reshape and normalize
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

image_noise = 20
rotation = 50
move = 4

epochs = 100000

min_lr = 0.00000001 #0.01
max_lr = 0.01 #0.25

# Initialize neural network and data lists
nn = NeuralNetwork(784, 1024, 10, load=True)

def augment_image(img_array, noise_multiplier, rotation_threshold, move_threshold, scale_threshold=0.30):
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

def full_test(batch_size=256):
    # --- Test set ---
    good = 0
    n_test = len(x_test)

    # random indices for test
    idxs = np.random.randint(0, n_test, size=n_test)

    # process in minibatches
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        batch_idxs = idxs[start:end]

        # build batch X of shape (B, D)
        X_batch = np.stack([np.asarray(x_test[i]).flatten() for i in batch_idxs], axis=0).astype(np.float32)

        # forward once for the whole batch
        out = nn.forward(X_batch)
        out = np.asarray(out)

        # get predictions: supports (B, C) or (C,) single-sample shape
        if out.ndim == 2:
            preds = np.argmax(out, axis=1)
        else:
            preds = np.argmax(out, axis=0)  # fallback

        # true labels
        trues = np.array([y_test[i] for i in batch_idxs])

        good += int(np.sum(preds == trues))

    accTest = round((good / n_test) * 10000) / 100

    # --- Train set (with augmentation) ---
    good = 0
    n_train = len(x_train)

    idxs = np.random.randint(0, n_train, size=n_train)

    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        batch_idxs = idxs[start:end]

        # apply augmentation per-sample but batch before forward
        batch_imgs = []
        for i in batch_idxs:
            batch_imgs.append(x_train[i].flatten())

        X_batch = np.stack(batch_imgs, axis=0).astype(np.float32)

        out = nn.forward(X_batch)
        out = np.asarray(out)
        if out.ndim == 2:
            preds = np.argmax(out, axis=1)
        else:
            preds = np.argmax(out, axis=0)

        trues = np.array([y_train[i] for i in batch_idxs])

        good += int(np.sum(preds == trues))

    accTrain = round((good / n_train) * 10000) / 100

    # --- Train set (with augmentation) ---
    good = 0
    n_train = len(x_train)

    idxs = np.random.randint(0, n_train, size=n_train)

    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        batch_idxs = idxs[start:end]

        # apply augmentation per-sample but batch before forward
        batch_imgs = []
        for i in batch_idxs:
            original = x_train[i]
            augmented = augment_image(original, image_noise, rotation, move)  # your augment params
            batch_imgs.append(np.asarray(augmented).flatten())

        X_batch = np.stack(batch_imgs, axis=0).astype(np.float32)

        out = nn.forward(X_batch)
        out = np.asarray(out)
        if out.ndim == 2:
            preds = np.argmax(out, axis=1)
        else:
            preds = np.argmax(out, axis=0)

        trues = np.array([y_train[i] for i in batch_idxs])

        good += int(np.sum(preds == trues))

    accAug = round((good / n_train) * 10000) / 100

    return accTest, accTrain, accAug

print("Testing...")

startTime = time.time()

testAcc, trainAcc, augAcc = full_test(batch_size=1024)

print(f"Test accuracy: {testAcc}\nTrain accuracy: {trainAcc}\nAugmented accuracy: {augAcc}")
print(f"Calculated in {round(time.time() - startTime, 4)} seconds")