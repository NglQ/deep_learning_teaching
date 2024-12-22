import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import random


# data preprocessing
def rescale_images_and_correct_masks( inputs):
  return {
    "images": tf.cast(inputs["image"], dtype=tf.float32) / 255.0,  # normalization
    "segmentation_masks": inputs["segmentation_mask"] - 1,  # put all values as 0-based.

  }
# now the label of the ground truth pixels are 0 for pet, 1 for borders, 2 for background


# utility function
def unpackage_inputs(inputs):
    images = inputs["images"]
    segmentation_masks = inputs["segmentation_masks"]
    return images, segmentation_masks


class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, val_ds, batch_size, epoch_interval=None, save_path=f'./model_weights/weights_img.weights.h5'):
        val_images, val_masks = next(iter(val_ds))
        self.epoch_interval = epoch_interval
        self.save_path = save_path
        self.val_images = val_images
        self.val_masks = val_masks
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_interval and epoch % self.epoch_interval == 0:

            self.model.save_weights(self.save_path)

            pred_masks = self.model.predict(self.val_images)
            pred_masks = tf.math.argmax(pred_masks, axis=-1)
            pred_masks = pred_masks[..., tf.newaxis] #add a new dimension at the end of pred_masks.
            # ... is a placeholder for dimensions

            # Randomly select an image from the test batch
            random_index = random.randint(0, self.batch_size - 1)
            random_image = self.val_images[random_index]
            random_pred_mask = pred_masks[random_index]
            random_true_mask = self.val_masks[random_index]

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
            ax[0].imshow(random_image)
            ax[0].set_title(f"Image: {epoch:03d}")

            ax[1].imshow(random_true_mask)
            ax[1].set_title(f"Ground Truth Mask: {epoch:03d}")

            ax[2].imshow(random_pred_mask)
            ax[2].set_title(f"Predicted Mask: {epoch:03d}", )

            plt.show()
            plt.close()


def plot_history(train_history, val_history, title):
    plt.plot(train_history)
    plt.plot(val_history)
    plt.grid()
    plt.xlabel('Epoch')
    plt.legend([title, f'Val {title}'])
    plt.title(title)
    plt.show()
    print()


def show_predicted_masks(test_ds, num_test_elements, model):
    test_images, test_masks = next(iter(test_ds))
    pred_masks = model.predict(test_images)
    pred_masks = tf.math.argmax(pred_masks, axis=-1)
    pred_masks = pred_masks[..., tf.newaxis]
    random_index = random.randint(0, 5)
    random_image = test_images[random_index]
    random_pred_mask = pred_masks[random_index]
    random_true_mask = test_masks[random_index]


    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    ax[0].imshow(random_image)
    ax[0].set_title(f"")

    ax[1].imshow(random_true_mask)
    ax[1].set_title(f"Ground Truth Mask")

    ax[2].imshow(random_pred_mask)
    ax[2].set_title(f"Predicted Mask")

    plt.show()
    plt.close()
