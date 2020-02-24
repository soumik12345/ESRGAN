import tensorflow as tf


def MultiStepLR(initial_learning_rate, lr_steps, lr_rate, name='MultiStepLR'):
    '''Multi-steps learning rate scheduler'''
    lr_steps_value = [initial_learning_rate]
    for _ in range(len(lr_steps)):
        lr_steps_value.append(lr_steps_value[-1] * lr_rate)
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=lr_steps, values=lr_steps_value
    )


def CosineAnnealingLR_Restart(initial_learning_rate, t_period, lr_min):
    '''Cosine annealing learning rate scheduler with restart'''
    return tf.keras.experimental.CosineDecayRestarts(
        initial_learning_rate=initial_learning_rate,
        first_decay_steps=t_period, t_mul=1.0, m_mul=1.0,
        alpha=lr_min / initial_learning_rate
    )