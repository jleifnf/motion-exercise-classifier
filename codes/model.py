"""

"""
from .preprocess import *
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D , MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """
    Generates data for Keras
    edited code from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, data, labels, batch_size=32, dim=(32, 32, 32),
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels  # if dict get the data
        self.data = data  # to pass in data
        # self.n_channels = n_channels  # infer n_channels from data
        # self.n_classes = n_classes  # infer from labels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        stride = 50  # samples = 1 second
        window = 250  # samples = 5 second for each observation
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.data[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, y


def batch_generator(data, targets=None, batchsize=100, specgram=False):
    """
    Generate batches data already split Test and Train by subjects.
    Args:
        data:
        targets: A list of str of exercises, or a dictionary of exercises with ('exercise','exercise_idx')
        batchsize:
        specgram:

    Returns:

    """

    # Data Preprocessing:
    if targets is None:
        targets = targets_idx
    elif isinstance(targets, list):
        targets.sort()
        targets = {ex: exercises[exercises.exercise == ex].index.to_list()[0] for ex in targets}

    # Data sanity check
    if isinstance(data, np.ndarray):
        # transform the data from ndarray to dictionary of exercises
        if data.ndim == 2 and data.shape[1] > len(targets):
            # filter out the data for just the target exercises
            target_dict = {e: data[:, c] for e, c in targets.items()}
            for k in target_dict:
                target_dict[k] = np.array([id for id in target_dict[k] if isinstance(id, dict)])
        elif data.ndim == 1:
            # assume the data is flattened into 1d and subjects vs exercises are shuffled
            # each index in the data is an individual subject with a different exercise
            target_dict = {e: data[:, c] for e, c in targets.items()}
            for k in target_dict:
                target_dict[k] = np.array([id for id in target_dict[k] if isinstance(id, dict)])

    sig = np.array()
    yield sig


def build_conv1d_model(n_classes, n_hidden=250, input_shape=(250,6)):
    model = Sequential()
    model.add(Conv1D(32, 5, input_shape = input_shape, activation='relu'))
    model.add(Conv1D(32, 5, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(16, 3, activation='relu'))
    model.add(Conv1D(16, 3, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(n_hidden, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def build_mlp_model(n_classes, hidden_layers=1, input_shape=None):
    init_nodes = 64

    model = Sequential()
    for i in range(hidden_layers):
        nodes = int(max(init_nodes / (2 * (i + 1)), 8))
        model.add(Dense(nodes, activation='softmax'))
        if nodes == 8:
            break
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
