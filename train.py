import logging
from src.utils import *
from src.losses import *
from src.models import *
from src.dataset import *
from src.training import *
from src.pretraining import *
from src.learning_rate_schedulers import *


class Trainer:

    def __init__(self, config_file):
        self.config = parse_config(config_file)
        logging.info('Setting Memory Growth')
        set_memory_growth()
        logging.info('Creating tfrecord file')
        self.create_tfrecord()
        logging.info('Preparing Dataset')
        self.dataset = self.get_dataset()
        logging.info('Creating Generator Model')
        self.generator = Generator(
            self.config['hr_patch_size'] // 4,
            self.config['n_channels']
        )
        logging.info('Creating Discriminator Model')
        self.discriminator = Discriminator(
            self.config['hr_patch_size'],
            self.config['n_channels']
        )
        logging.info('Pretraining Generator with Pixel Loss')
        (
            self.pretrain_checkpoint,
            self.pretrain_checkpoint_manager,
            self.generator,
            self.pretrain_gen_optimizer
        ) = self.pre_train()
        logging.info('Training Generator and Discriminator')
        (
            self.checkpoint,
            self.checkpoint_manager,
            self.generator,
            self.discriminator,
            self.optimizer_G,
            self.optimizer_D
        ) = self.train()

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

    def train(self):
        gen_lr = MultiStepLR(
            float(self.config['train']['generator']['lr_scheduler']['initial_learning_rate']),
            self.config['train']['generator']['lr_scheduler']['lr_steps'],
            self.config['train']['generator']['lr_scheduler']['lr_rate']
        )
        dis_lr = MultiStepLR(
            float(self.config['train']['discriminator']['lr_scheduler']['initial_learning_rate']),
            self.config['train']['discriminator']['lr_scheduler']['lr_steps'],
            self.config['train']['discriminator']['lr_scheduler']['lr_rate']
        )
        optimizer_G = tf.keras.optimizers.Adam(
            learning_rate=gen_lr,
            beta_1=self.config['train']['generator']['optimizer']['beta_1'],
            beta_2=self.config['train']['generator']['optimizer']['beta_2']
        )
        optimizer_D = tf.keras.optimizers.Adam(
            learning_rate=gen_lr,
            beta_1=self.config['train']['discriminator']['optimizer']['beta_1'],
            beta_2=self.config['train']['discriminator']['optimizer']['beta_2']
        )
        pixel_loss_fn = PixelLoss()
        fea_loss_fn = ContentLoss()
        gen_loss_fn = GeneratorLoss()
        dis_loss_fn = DiscriminatorLoss()
        checkpoint, checkpoint_manager, generator, discriminator, optimizer_G, optimizer_D = train(
            self.dataset, [self.generator, self.discriminator],
            [optimizer_G, optimizer_D],
            [pixel_loss_fn, fea_loss_fn, gen_loss_fn, dis_loss_fn],
            self.config['train']['epochs'],
            save_interval=self.config['train']['save_interval'],
            checkpoint_dir=self.config['train']['checkpoint_dir'],
            log_dir=self.config['train']['log_dir']
        )
        return checkpoint, checkpoint_manager, generator, discriminator, optimizer_G, optimizer_D
