import numpy as np
from .utils import *
import tensorflow as tf



def infer_on_patch(image_patch, model):
    return model(image_patch)


def infer_on_patches(image_file, model, patch_size, stride):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    predicted_image = tf.ones_like(image)
    i, patches = 0, []
    height, width, _ = image.shape
    while i < width:
        j = 0
        while j < height:
            patch = image[j:patch_size + j, i:patch_size + i, :]
            patch = patch * 2.0 - 1.0
            patch = tf.expand_dims(patch, axis=0)
            if patch.shape == (1, patch_size, patch_size, 3):
                patches.append(model(patch))
            j += stride
        i += stride
    return patches