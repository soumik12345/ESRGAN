import os
import numpy as np
from PIL import Image
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
    return tf.cast((tensor + 1.0) / 2.0, tf.uint8)


def denormalize_prediction(tensor):
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
                ax.imshow(denormalize(x_batch[c]))
                ax.set_xlabel('Low_Res_' + str(c + 1))
            elif i % 2 == 1:
                ax.imshow(denormalize(y_batch[c]))
                ax.set_xlabel('High_Res_' + str(c + 1))
                c += 1
    else:
        fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (16, 16))
        plt.setp(axes.flat, xticks = [], yticks = [])
        for i, ax in enumerate(axes.flat):
            if i % 3 == 0:
                ax.imshow(denormalize(x_batch[c]))
                ax.set_xlabel('Low_Res_' + str(c + 1))
            elif i % 3 == 1:
                ax.imshow(denormalize(y_batch[c]))
                ax.set_xlabel('High_Res_' + str(c + 1))
            elif i % 3 == 2:
                pred = np.squeeze(model(np.expand_dims(x_batch[c], axis=0)))
                ax.imshow(denormalize_prediction(pred))
                ax.set_xlabel('High_Res_' + str(c + 1))
                c += 1
    plt.show()



def network_interpolation(model, pretrain_ckpt_path, train_ckpt_path, alpha):
    
    def update_weight(model, vars1, vars2, alpha):
        for i, var in enumerate(model.trainable_variables):
            var.assign((1 - alpha) * vars1[i] + alpha * vars2[i])
        return model
    
    ckpt_1 = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(pretrain_ckpt_path):
        ckpt_1.restore(tf.train.latest_checkpoint(pretrain_ckpt_path))
    else:
        print('Cannot find checkpoint')
    
    ckpt_2 = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(train_ckpt_path):
        ckpt_2.restore(tf.train.latest_checkpoint(train_ckpt_path))
    else:
        print('Cannot find checkpoint')
    
    variables_1 = [v.numpy() for v in ckpt_1.model.trainable_variables]
    variables_2 = [v.numpy() for v in ckpt_2.model.trainable_variables]

    return update_weight(model, variables_1, variables_2, alpha)



def save_all_crops(image_file, cache_location, patch_size, stride):
    image = Image.open(image_file)
    image = np.array(image)
    height, width, _ = image.shape
    i, c = 0, 0
    save_location = os.path.join(cache_location, image_file[:-4])
    print(save_location)
    try:
        os.mkdir(save_location)
    except:
        pass
    while i < width:
        j = 0
        while j < height:
            patch = np.asarray(image[j:patch_size + j, i:patch_size + i, :])
            if patch.shape == (patch_size, patch_size, 3):
                patch = Image.fromarray(patch)
                c += 1
                patch.save(os.path.join(save_location, str(c) + '.png'))
            j += stride
        i += stride



def get_all_crops(image, patch_size, stride):
    height, width, _ = image.shape
    i = 0
    patches = []
    while i < width:
        j = 0
        while j < height:
            patch = np.asarray(image[j:patch_size + j, i:patch_size + i, :])
            if patch.shape == (patch_size, patch_size, 3):
                patches.append(patch)
            j += stride
        i += stride
    return patches
