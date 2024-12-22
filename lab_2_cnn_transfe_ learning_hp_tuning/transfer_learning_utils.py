import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, GlobalAveragePooling2D
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


def get_model(input_shape, num_classes, dropout_rate=.0, data_augmentation=None,
              activation='relu', padding='same', kernel_size=3):
    x = Input(shape=input_shape)

    if data_augmentation:
        x = (data_augmentation(input_shape))(x)

    r = Rescaling(1. / 255.)(x)

    c1 = Conv2D(16, kernel_size, padding=padding, activation=activation)(r)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, kernel_size, padding=padding, activation=activation)(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(64, kernel_size, padding=padding, activation=activation)(p2)
    p3 = MaxPooling2D((2, 2))(c3)

    if dropout_rate > .0:
        p3 = Dropout(dropout_rate)(p3)

    f = Flatten()(p3)
    d1 = Dense(128, activation=activation)(f)
    d2 = Dense(num_classes)(d1)

    model = Model(x, d2)

    return model


def get_pre_trained_model(input_shape, num_classes, base_model):

    data_augmentation = lambda input_shape: tf.keras.Sequential([
        RandomFlip("horizontal", input_shape=input_shape),
        RandomRotation(0.1),
        RandomZoom(0.1),
    ])

    input = tf.keras.Input(shape=input_shape)
    aug = (data_augmentation(input_shape))(input)
    r = Rescaling(1./127.5, offset=-1)(aug)  # MobileNetV2 expects inputs in the range [-1, 1]
    m_n = base_model(r, training=False)
    gap = GlobalAveragePooling2D()(m_n)
    d = Dropout(0.2)(gap)
    out = Dense(num_classes)(d)
    model = tf.keras.Model(input, out)

    return model


def display_history(history):
    mse_training = history.history['loss']
    acc_training = history.history['accuracy']

    mse_val = history.history['val_loss']
    acc_val = history.history['val_accuracy']

    # Visualize the behavior of the loss
    plt.plot(mse_training)
    plt.plot(mse_val)
    plt.grid()
    plt.title('Loss during training')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'])
    plt.show()

    # and of the accuracy
    plt.plot(acc_training)
    plt.plot(acc_val)
    plt.grid()
    plt.title('Accuracy during training')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'])
    plt.show()


def show_images(train_ds, class_names, data_augmentation=None):
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            image_to_show = data_augmentation(images)[0] if data_augmentation else images[i]
            plt.imshow(image_to_show.numpy().astype("uint8"))
            if not data_augmentation:
                plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()


def append_history(history_pre_train, history_fine_tune):
    history_pre_train.history['accuracy'] += history_fine_tune.history['accuracy']
    history_pre_train.history['val_accuracy'] += history_fine_tune.history['val_accuracy']

    history_pre_train.history['loss'] += history_fine_tune.history['loss']
    history_pre_train.history['val_loss'] += history_fine_tune.history['val_loss']
