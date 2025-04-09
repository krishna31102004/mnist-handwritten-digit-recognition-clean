# load_mnist.py
import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def load_and_preprocess():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("Original training data shape:", x_train.shape)   # (60000, 28, 28)
    print("Original test data shape:", x_test.shape)         # (10000, 28, 28)

    # Normalize the data (scale pixel values to range [0, 1])
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0

    # Flatten the data for traditional ML models (28x28 -> 784)
    x_train_flat = x_train.reshape(x_train.shape[0], 28*28)
    x_test_flat  = x_test.reshape(x_test.shape[0], 28*28)

    print("Flattened training data shape:", x_train_flat.shape)  # (60000, 784)
    print("Flattened test data shape:", x_test_flat.shape)       # (10000, 784)
    
    # Visualize a few sample images (to verify data quality)
    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(x_train[i], cmap='gray')
        plt.title(f"Label: {y_train[i]}")
        plt.axis('off')
    plt.show()
    
    return (x_train, y_train, x_test, y_test, x_train_flat, x_test_flat)
    
if __name__ == '__main__':
    load_and_preprocess()
