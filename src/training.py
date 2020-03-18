from tqdm import tqdm
from .utils import *
import tensorflow as tf


def get_checkpoint_train(models, optimizers, checkpoint_dir='./train_checkpoints/'):
    '''Get Training Checkpoints
    Params:
        models          -> [generator, discriminator]
        optimizers      -> [gen_optimizer, dis_optimizer]
        checkpoint_dir  -> Checkpoint Directory
    '''
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0, name='step'),
        gen_model=models[0], dis_model=models[1],
        gen_optimizer=optimizers[0], dis_optimizer=optimizers[1]
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=checkpoint_dir,
        max_to_keep=5
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


def train(
        dataset, models, optimizers, loss_functions, epochs, save_interval=5,
        checkpoint_dir='./models/pretrain/', log_dir='./logs/train'):
    '''ESRGAN Train
    Params:
        dataset         -> Dataset Object
        models          -> [generator, discriminator]
        optimizers      -> [gen_optimizer, dis_optimizer]
        loss_functions  -> [pixel_loss_fn, fea_loss_fn, gen_loss_fn, dis_loss_fn]
        epochs          -> Number of epochs
        checkpoint_dir  -> Checkpoint Directory
        log_dir         -> Tensorboard Log Directory
    '''
    writer = tf.summary.create_file_writer(log_dir)
    checkpoint, checkpoint_manager = get_checkpoint_train(
        models, optimizers,
        checkpoint_dir=checkpoint_dir
    )
    generator, discriminator = models
    optimizer_G, optimizer_D = optimizers
    pixel_loss_fn, fea_loss_fn, gen_loss_fn, dis_loss_fn = loss_functions

    @tf.function
    def train_step(lr, hr):
        with tf.GradientTape(persistent=True) as tape:
            sr = generator(lr, training=True)
            hr_output = discriminator(hr, training=True)
            sr_output = discriminator(sr, training=True)
            losses_G = {}
            losses_D = {}
            losses_G['regular'] = tf.reduce_sum(generator.losses)
            losses_D['regular'] = tf.reduce_sum(discriminator.losses)
            losses_G['pixel'] = float(1e-2) * pixel_loss_fn(hr, sr)
            losses_G['feature'] = 1.0 * fea_loss_fn(hr, sr)
            losses_G['gan'] = float(5e-3) * gen_loss_fn(hr_output, sr_output)
            losses_D['gan'] = dis_loss_fn(hr_output, sr_output)
            total_loss_G = tf.add_n([l for l in losses_G.values()])
            total_loss_D = tf.add_n([l for l in losses_D.values()])
        grads_G = tape.gradient(
            total_loss_G,
            generator.trainable_variables
        )
        grads_D = tape.gradient(
            total_loss_D,
            discriminator.trainable_variables
        )
        optimizer_G.apply_gradients(
            zip(
                grads_G,
                generator.trainable_variables
            )
        )
        optimizer_D.apply_gradients(
            zip(
                grads_D,
                discriminator.trainable_variables
            )
        )
        return total_loss_G, total_loss_D, losses_G, losses_D

    steps = 0
    for epoch in range(1, epochs + 1):
        print('Epoch: {}'.format(epoch))
        for (batch, (lr, hr)) in tqdm(enumerate(dataset.take(1000000))):
            steps += 1
            total_loss_G, total_loss_D, losses_G, losses_D = train_step(lr, hr)
            with writer.as_default():
                tf.summary.scalar('loss_G/total_loss', total_loss_G.numpy(), step=steps)
                tf.summary.scalar('loss_D/total_loss', total_loss_D.numpy(), step=steps)
                for k, l in losses_G.items():
                    tf.summary.scalar('loss_G/{}'.format(k), l.numpy(), step=steps)
                for k, l in losses_D.items():
                    tf.summary.scalar('loss_D/{}'.format(k), l.numpy(), step=steps)
                tf.summary.scalar('learning_rate/learning_rate_G', optimizer_G.lr(steps), step=steps)
                tf.summary.scalar('learning_rate/learning_rate_D', optimizer_D.lr(steps), step=steps)
                tf.summary.image('Low Res', denormalize(lr), step=steps)
                tf.summary.image('High Res', denormalize(hr), step=steps)
                tf.summary.image('Generated', denormalize_prediction(generator(lr)), step=steps)
        if epoch % save_interval == 0:
            checkpoint_manager.save()
            print(
                'Model Checkpoints saved at {}'.format(
                    checkpoint_manager.latest_checkpoint
                )
            )

    return checkpoint, checkpoint_manager, generator, discriminator, optimizer_G, optimizer_D
