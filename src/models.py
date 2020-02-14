from .blocks import *
import tensorflow as tf



def Generator(size, channels, cfg_net, gc=32, wd=0., name='Generator'):
    '''Generator Model'''
    nf, nb = cfg_net['nf'], cfg_net['nb']
    lrelu_f = functools.partial(tf.keras.layers.LeakyReLU, alpha=0.2)
    rrdb_f = functools.partial(ResInResDenseBlock, nf=nf, gc=gc, wd=wd)
    conv_f = functools.partial(
        tf.keras.layers.Conv2D, kernel_size=3, padding='same',
        bias_initializer='zeros', kernel_initializer=_kernel_init(),
        kernel_regularizer=_regularizer(wd)
    )
    rrdb_truck_f = tf.keras.Sequential(
        [rrdb_f(name="RRDB_{}".format(i)) for i in range(nb)],
        name='RRDB_trunk'
    )
    # extraction
    x = inputs = tf.keras.Input(
        [size, size, channels],
        name='input_image'
    )
    fea = conv_f(filters=nf, name='conv_first')(x)
    fea_rrdb = rrdb_truck_f(fea)
    trunck = conv_f(filters=nf, name='conv_trunk')(fea_rrdb)
    fea = fea + trunck
    # upsampling
    size_fea_h = tf.shape(fea)[1] if size is None else size
    size_fea_w = tf.shape(fea)[2] if size is None else size
    fea_resize = tf.image.resize(
        fea, [size_fea_h * 2, size_fea_w * 2],
        method='nearest', name='upsample_nn_1'
    )
    fea = conv_f(filters=nf, activation=lrelu_f(), name='upconv_1')(fea_resize)
    fea_resize = tf.image.resize(
        fea, [size_fea_h * 4, size_fea_w * 4],
        method='nearest', name='upsample_nn_2'
    )
    fea = conv_f(filters=nf, activation=lrelu_f(), name='upconv_2')(fea_resize)
    fea = conv_f(filters=nf, activation=lrelu_f(), name='conv_hr')(fea)
    out = conv_f(filters=channels, name='conv_last')(fea)
    return tf.keras.Model(inputs, out, name=name)