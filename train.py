import tensorflow as tf
from src.utils import *
from src.losses import *
from src.models import *
from src.dataset import *
from src.pretraining import *
from src.learning_rate_schedulers import *



class Trainer:

    def __init__(self, config_file):
        self.config = parse_config(config_file)
        set_memory_growth()
        self.create_tfrecord()
        self.dataset = self.get_dataset()
        self.generator = Generator(
            self.config['hr_patch_size'] / 4,
            self.config['n_channels']
        )
        self.discriminator = Discriminator(
            self.config['hr_patch_size'],
            self.config['n_channels']
        )
        (
            self.pretrain_checkpoint,
            self.pretrain_checkpoint_manager,
            self.generator,
            self.pretrain_gen_optimizer
        ) = self.pre_train()

    def create_tfrecord(self):
        tfrecord_creator = TFRecordCreator(
            self.config['hr_img_path'][:-1],
            self.config['lr_img_path'][:-1]
        )
        tfrecord_creator.make_tfrecord_file(self.config['tfrecord_file'])

    def get_dataset(self):
        dataloader = SRTfrecordDataset(self.config['hr_patch_size'])
        dataset = dataloader.get_dataset(
            self.config['tfrecord_file'],
            self.config['batch_size'],
            self.config['buffer_size']
        )
        return dataset

    def pre_train(self):
        learning_rate = MultiStepLR(
            float(self.config['pretrain']['lr_schedulers_pretrain']['initial_learning_rate']),
            self.config['pretrain']['lr_schedulers_pretrain']['lr_steps'],
            self.config['pretrain']['lr_schedulers_pretrain']['lr_rate']
        )
        gen_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=self.config['pretrain']['optimizer']['beta_1'],
            beta_2=self.config['pretrain']['optimizer']['beta_2']
        )
        pixel_loss = PixelLoss()
        checkpoint, checkpoint_manager, generator, gen_optimizer = pretrain(
            self.dataset, self.generator, gen_optimizer, pixel_loss,
            checkpoint_dir=self.config['pretrain']['checkpoint_dir'],
            log_dir=self.config['pretrain']['log_dir']
        )
        return checkpoint, checkpoint_manager, generator, gen_optimizer