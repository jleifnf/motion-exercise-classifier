"""
Building Keras LSTM model for automatic recognition of fitness motion.
Dependencies:
- Keras  ->  2.3.1
"""
from .preprocess import *
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D , MaxPooling1D, LSTM
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras.utils import Sequence


n_hidden = 250
epochs = 25
checkpoint_file = os.path.join('./training_checkpoints', f"motions_LSTM_v1"+".h5")

checkpoint_callback= ModelCheckpoint(
                                monitor='val_accuracy',
                                save_best_only=True,
                                filepath=checkpoint_file,
                                save_weights_only=False,
                                period=2)


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


def build_lstm_model(n_classes, n_hidden=250, input_shape=(250,6)):
    model = Sequential()
    model.add(Conv1D(32, 5, input_shape = input_shape))
    model.add(Conv1D(32, 5))
    model.add(MaxPooling1D())
    model.add(Conv1D(16, 3))
    model.add(Conv1D(16, 3))
    model.add(MaxPooling1D())
    model.add(Conv1D(8, 3))
    model.add(Conv1D(8, 3))
    model.add(MaxPooling1D())
#     model.add(Flatten())
#     model.add(LSTM(n_hidden))
    model.add(LSTM(n_hidden, dropout=0.1))
#     model.add(Dense(n_hidden, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
