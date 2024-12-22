from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dropout
import matplotlib.pyplot as plt


def get_model(input_shape, regularizer=None, dropout=False):  # regularizers.l2(0.005)
    seq = Input(shape=input_shape)
    x = Dense(64, activation='relu', kernel_regularizer=regularizer)(seq)

    if dropout:
        x = Dropout(0.5)(x)
    
    x = Dense(16, activation='relu', kernel_regularizer=regularizer)(x)

    if dropout:
        x = Dropout(0.5)(x)

    x = Dense(1, activation='sigmoid')(x)

    return Model(seq, x)


def plot_history(model_history,keys):
    m,val_m = keys
    plt.plot(model_history.history[m])
    plt.plot(model_history.history[val_m])
    plt.ylabel(m)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

