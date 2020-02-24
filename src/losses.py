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



def ContentLoss(type='l1', output_layer=54, before_act=True):
    '''Content Loss (VGG19)
    Params:
        type        -> 'l1'/'l2' norm
        output_layer    -> Index of VGG19 output Layer
        before_act      -> Include activation or not
    '''
    if type == 'l1':
        loss_function = tf.keras.losses.MeanAbsoluteError()
    else:
        loss_function = tf.keras.losses.MeanSquaredError()
    vgg = tf.keras.applications.VGG19(
        input_shape=(None, None, 3),
        include_top=False
    )
    if output_layer == 22:
        pick_layer = 5
    elif output_layer == 54:
        pick_layer = 20
    if before_act:
        vgg.layers[pick_layer].activation = None
    feature_extration_model = tf.keras.Model(
        vgg.input,
        vgg.layers[pick_layer].output
    )

    @tf.function
    def content_loss(hr, sr):
        preprocess_sr = tf.keras.applications.vgg19.preprocess_input(sr * 255.) / 12.75
        preprocess_hr = tf.keras.applications.vgg19.preprocess_input(hr * 255.) / 12.75
        sr_features = feature_extration_model(preprocess_sr)
        hr_features = feature_extration_model(preprocess_hr)
        return loss_function(hr_features, sr_features)
    
    return content_loss