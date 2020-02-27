import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print("Detect {} Physical GPUs, {} Logical GPUs.".format(len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def denormalize(tensor):
    return tf.cast(255 * (tensor + 1.0) / 2.0, tf.uint8)
            
            
def visualize_batch(dataset, model=None):
    x_batch, y_batch = next(iter(dataset))
    x_batch = x_batch.numpy()
    y_batch = y_batch.numpy()
    c = 0
    if model is None:
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 16))
        plt.setp(axes.flat, xticks = [], yticks = [])
        for i, ax in enumerate(axes.flat):
            if i % 2 == 0:
                ax.imshow(x_batch[c])
                ax.set_xlabel('Low_Res_' + str(c + 1))
            elif i % 2 == 1:
                ax.imshow(y_batch[c])
                ax.set_xlabel('High_Res_' + str(c + 1))
                c += 1
    else:
        fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (16, 16))
        plt.setp(axes.flat, xticks = [], yticks = [])
        for i, ax in enumerate(axes.flat):
            if i % 3 == 0:
                ax.imshow(x_batch[c])
                ax.set_xlabel('Low_Res_' + str(c + 1))
            elif i % 3 == 1:
                ax.imshow(y_batch[c])
                ax.set_xlabel('High_Res_' + str(c + 1))
            elif i % 3 == 2:
                pred = np.squeeze(model(np.expand_dims(x_batch[c], axis=0)))
                ax.imshow(denormalize(pred))
                ax.set_xlabel('High_Res_' + str(c + 1))
                c += 1
    plt.show()