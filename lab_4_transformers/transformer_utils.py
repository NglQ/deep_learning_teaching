import tensorflow as tf
import re
import string
from matplotlib import pyplot as plt

def custom_preprocessing(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    no_punctuation = tf.strings.regex_replace(stripped_html, f"[{re.escape(string.punctuation)}]", "")
    return no_punctuation


def plot_history(train_history, val_history, title):
    plt.plot(train_history)
    plt.plot(val_history)
    plt.grid()
    plt.xlabel('Epoch')
    plt.legend([title, f'Val {title}'])
    plt.title(title)
    plt.show()
    print()
