import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import importlib
import keras_tuner as kt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, MaxPooling2D, Conv2D

from sklearn.model_selection import train_test_split, KFold

import sys
import os

if len(sys.argv) != 2:
    print("Missing model path")
    exit()

model_path = sys.argv[-1]

if not os.path.exists(model_path):
    print("Model path does not exist")
    exit()


def obtain_model(model_path):
    try:
        model = getattr(importlib.import_module(model_path.replace('/', '.').replace(".py", "")), "mnist_cnn")
        config = getattr(importlib.import_module(model_path.replace('/', '.').replace(".py", "")), "config")

        return model, config
    except ModuleNotFoundError:
        print("Model path does not exist")
        exit()


# Obtain the data from the mnist set
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Convert ground-truth to categorical labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Recombine into one set for splitting via KFold
X = np.vstack([X_train, X_test])
y = np.vstack([y_train, y_test])

# Normalize the data
X = X / 255

# We'll use array indices to distinguish between folds. For now, it's the simplest to read and code.
folds = 10
amount_per_fold = len(X) / folds

fold_indices = []
for i in range(0, folds):
    fold_indices.append((int(i * amount_per_fold), int((i + 1) * amount_per_fold - 1)))

# Now, fold_indices contains the range of each fold
# For example, fold_indices[3] is the fourth fold.

test_accuracies = []
for i in range(0, folds):
    print("Determining validation set...")
    valid_folds = set(range(0, folds)) - set([i])
    validation_fold = np.random.randint(0, len(valid_folds))

    print("Building training set...")
    valid_folds -= set([validation_fold])
    valid_folds = list(valid_folds)

    X_train = None
    y_train = None
    for j in valid_folds:
        start, end = fold_indices[j]

        if X_train is None and y_train is None:
            X_train = np.array(X[start: end + 1])
            y_train = np.array(y[start: end + 1])
        else:
            X_train = np.vstack([X_train, X[start: end + 1]])
            y_train = np.vstack([y_train, y[start: end + 1]])

    # Now, both X_train and y_train have size (folds - 2) * amount_per_fold

    print("Preparing validation set...")
    start, end = fold_indices[validation_fold]
    X_val = np.array(X[start: end + 1])
    y_val = np.array(y[start: end + 1])

    print("Preparing test set...")
    start, end = fold_indices[i]
    X_test = np.array(X[start: end + 1])
    y_test = np.array(y[start: end + 1])

    print("Initializing model %d..." % (i + 1))
    model, config = obtain_model(model_path)

    tuner = kt.RandomSearch(
        model.initialize,
        objective='val_loss',
        max_trials=50
    )

    tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    model = tuner.get_best_models()[0]

    # best_hyperparameters = tuner.get_best_hyperparameters()[0]
    # model = model.initialize(best_hyperparameters)
    print(model.summary())
    # print(tuner.results_summary())
    exit()

    '''
    history = model.fit(
            X_train,
            y_train,
            validation_data = (X_val, y_val),
            epochs = config.epochs,
            batch_size = config.batch_size,
            workers = config.workers,
    ).history
    '''

    test_accuracies.append(model.evaluate(X_test, y_test)[1])
    print("*** This fold gave an accuracy of ", test_accuracies[-1])
    plt.plot(history['accuracy'], color='blue')
    plt.plot(history['val_accuracy'], color='red')

'''
print("Average model accuracy: ", np.average(test_accuracies))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(range(0, config.epochs, int(config.epochs / 10)))
plt.gca().legend(("Training", "Validation"))
plt.show()
'''

# Notes:
# TODO
#   Better graphs?
#   Confusion Matrix could be cool, might not work here

# TO TEST
# Validation sets? Idk
# Batch size
# Number of filters
# Presence of max pooling
# Normalizing input data
# More convolution layers
# Bigger filters
# Strides
# Kernel sizes (Bigger conv layers)
# BatchNorm
# More denses
# Purely convolution network (no flatten)
# Activation functions
# padding
# Epochs
# workers? (In fit)
