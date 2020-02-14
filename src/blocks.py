import functools
import tensorflow as tf


def kernel_regularizer(weight_decay=5e-4):
    '''L2 Weight Decay
    Params:
        weight_decay -> Weight Decay
    '''
    return tf.keras.regularizers.l2(weight_decay)


def kernel_initializer(scale=1.0, seed=None):
    '''He normal initializer with scale'''
    scale = 2. * scale
    return tf.keras.initializers.VarianceScaling(
        scale=scale, mode='fan_in',
        distribution="truncated_normal", seed=seed
    )


class BatchNormalization(tf.keras.layers.BatchNormalization):
    '''Make trainable=False freeze BN for real (the og version is sad).
    Reference: https://github.com/zzh8829/yolov3-tf2
    '''
    
    def __init__(
        self, axis=-1, momentum=0.9,
        epsilon=1e-5, center=True,
        scale=True, name=None, **kwargs):
        
        super(BatchNormalization, self).__init__(
            axis=axis, momentum=momentum,
            epsilon=epsilon, center=center,
            scale=scale, name=name, **kwargs
        )

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class ResDenseBlock5C(tf.keras.layers.Layer):
    '''Residual Dense Block'''
    
    def __init__(
        self, nf=64, gc=32, res_beta=0.2,
        wd=0., name='RDB5C', **kwargs):
        
        super(ResDenseBlock5C, self).__init__(name=name, **kwargs)
        self.res_beta = res_beta
        lrelu_f = functools.partial(tf.keras.layers.LeakyReLU, alpha=0.2)
        _Conv2DLayer = functools.partial(
            tf.keras.layers.Conv2D, kernel_size=3,
            padding='same', kernel_initializer=kernel_initializer(0.1),
            bias_initializer='zeros', kernel_regularizer=kernel_regularizer(wd)
        )
        self.conv1 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv2 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv3 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv4 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv5 = _Conv2DLayer(filters=nf, activation=lrelu_f())

    def call(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(tf.concat([x, x1], 3))
        x3 = self.conv3(tf.concat([x, x1, x2], 3))
        x4 = self.conv4(tf.concat([x, x1, x2, x3], 3))
        x5 = self.conv5(tf.concat([x, x1, x2, x3, x4], 3))
        return x5 * self.res_beta + x


class ResInResDenseBlock(tf.keras.layers.Layer):
    '''Residual in Residual Dense Block'''
    
    def __init__(
        self, nf=64, gc=32, res_beta=0.2,
        wd=0., name='RRDB', **kwargs):
        super(ResInResDenseBlock, self).__init__(name=name, **kwargs)
        self.res_beta = res_beta
        self.rdb_1 = ResDenseBlock5C(nf, gc, res_beta=res_beta, wd=wd)
        self.rdb_2 = ResDenseBlock5C(nf, gc, res_beta=res_beta, wd=wd)
        self.rdb_3 = ResDenseBlock5C(nf, gc, res_beta=res_beta, wd=wd)

    def call(self, x):
        out = self.rdb_1(x)
        out = self.rdb_2(out)
        out = self.rdb_3(out)
        return out * self.res_beta + x