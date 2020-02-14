import tensorflow as tf


def GeneratorLoss(type='relativistic'):
    '''Generator Loss
    Params:
        type -> 'relativistic'/'vanilla'
    '''
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    sigma = tf.sigmoid

    def generator_loss_relativistic(hr, sr):
        return 0.5 * (
            bce(tf.ones_like(sr), sigma(sr - tf.reduce_mean(hr))) +
            bce(tf.zeros_like(hr), sigma(hr - tf.reduce_mean(sr)))
        )

    def generator_loss_vanilla(hr, sr):
        return bce(tf.ones_like(sr), sigma(sr))

    if type == 'relativistic':
        return generator_loss_relativistic
    else:
        return generator_loss_vanilla