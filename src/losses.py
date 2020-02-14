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



def DiscriminatorLoss(type='relativistic'):
    '''Generator Loss
    Params:
        type -> 'relativistic'/'vanilla'
    '''
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    sigma = tf.sigmoid

    def discriminator_loss_relativistic(hr, sr):
        return 0.5 * (
            bce(tf.ones_like(hr), sigma(hr - tf.reduce_mean(sr))) +
            bce(tf.zeros_like(sr), sigma(sr - tf.reduce_mean(hr)))
        )

    def discriminator_loss_vanilla(hr, sr):
        real_loss = bce(tf.ones_like(hr), sigma(hr))
        fake_loss = bce(tf.zeros_like(sr), sigma(sr))
        return real_loss + fake_loss
    
    if type == 'relativistic':
        return discriminator_loss_relativistic
    else:
        return discriminator_loss_vanilla



def PixelLoss(type='l1'):
    '''Pixel Loss
    Reference:
        type -> 'l1'/'l2' norm
    '''
    if type == 'l1':
        return tf.keras.losses.MeanAbsoluteError()
    else:
        return tf.keras.losses.MeanSquaredError()