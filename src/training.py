import tensorflow as tf



def get_checkpoint_train(models, optimizers, checkpoint_dir='./checkpoints/'):
    '''Get Training Checkpoints
    Params:
        models          -> [generator, discriminator]
        optimizers      -> [gen_optimizer, dis_optimizer]
        checkpoint_dir  -> Checkpoint Directory
        
    '''
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0, name='step'),
        gen_model = models[0], dis_model = models[1],
        gen_optimizer=optimizers[0], dis_optimizer=optimizers[1]
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=checkpoint_dir,
        max_to_keep=3
    )
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(
            'Loaded ckpt from {} at step {}.'.format(
                checkpoint_manager.latest_checkpoint,
                checkpoint.step.numpy()
            )
        )
    else:
        print("Training from scratch....")
    return checkpoint, checkpoint_manager