from tqdm import tqdm
import tensorflow as tf



def get_checkpoint_pretrain(model, optimizer, checkpoint_dir='./checkpoints/'):
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0, name='step'),
        optimizer=optimizer, model=model
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



def pretrain(
    dataset, generator, optimizer, pixel_loss,
    checkpoint_dir='./models/pretrain/', log_dir='./logs/pretrain'):
    '''Generator Pretrain
    Params:
        dataset             -> Dataset Object
        generator           -> Generator Model
        optimizer           -> Generator Optimizer
        pizel_loss          -> Pixel Loss Function
        checkpoint          -> Checkpoint Object
        checkpoint_manager  -> Checkpoint Manager Object
        log_dir             -> Tensorboard Log Directory
    '''
    writer = tf.summary.create_file_writer(log_dir)
    checkpoint, checkpoint_manager = get_checkpoint(
        generator, optimizer,
        checkpoint_dir=checkpoint_dir
    )

    @tf.function
    def train_step(lr, hr):
        with tf.GradientTape() as tape:
            sr = generator(lr, training=True)
            losses = {}
            losses['regular'] = tf.reduce_sum(generator.losses)
            losses['pixel'] = 1.0 * pixel_loss(hr, sr)
            total_loss = tf.add_n([l for l in losses.values()])
        grads = tape.gradient(total_loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        return total_loss, losses

    for (batch, (lr, hr)) in tqdm(enumerate(dataset)):
        total_loss, losses = train_step(lr, hr)
        with writer.as_default():
            tf.summary.scalar('loss/total_loss', total_loss, step=batch)
            for k, l in losses.items():
                tf.summary.scalar('loss/{}'.format(k), l, step=batch)
            tf.summary.scalar('learning_rate', optimizer.lr(batch), step=batch)
    
    checkpoint_manager.save()
    print(
        'Pre-trained Generator saved at {}'.format(
            checkpoint_manager.latest_checkpoint
        )
    )

    return checkpoint, checkpoint_manager, generator, optimizer