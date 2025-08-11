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

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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

def full_test(nn):
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

print("Testing...")

startTime = time.time()

testAcc, trainAcc = full_test(nn)

print(f"Test accuracy: {testAcc}\nTrain accuracy: {trainAcc}")
print(f"Calculated in {round(time.time() - startTime, 4)} seconds")