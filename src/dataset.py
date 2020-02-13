import tensorflow as tf


class SRTfrecordDataset:

    def __init__(self, gt_size, scale=4, apply_flip=True, apply_rotation=True):
        self.gt_size = gt_size
        self.scale = scale
        self.apply_flip = apply_flip
        self.apply_rotation = apply_rotation
    
    def apply_random_crop(self, lr_image, hr_image):
        lr_image_shape = tf.shape(lr_image)
        hr_image_shape = tf.shape(hr_image)
        gt_shape = (
            self.gt_size,
            self.gt_size,
            hr_image_shape[-1]
        )
        lr_shape = (
            self.gt_size // self.scale,
            self.gt_size // self.scale,
            lr_image_shape[-1]
        )
        limit - lr_image_shape - lr_shape + 1
        offset = tf.random.uniform(
            tf.shape(lr_image_shape),
            dtype=tf.int32, maxval=tf.int32.max
        ) % limit
        lr_image = tf.slice(lr_image, offset, lr_shape)
        hr_img = tf.slice(hr_image, offset * scale, gt_shape)
        return lr_image, hr_image
    
    def apply_flip_to_images(self, lr_image, hr_image):
        def _flip():
            return (
                tf.image.flip_left_right(lr_image),
                tf.image.flip_left_right(hr_image)
            )
        flip_case = tf.random.uniform([1], 0, 2, dtype=tf.int32)
        lr_image, hr_image = tf.case(
            [(tf.equal(flip_case, 0), _flip)],
            default=lambda: (lr_image, hr_image)
        )
        return lr_image, hr_image
    
    def apply_rotation_to_images(self, lr_image, hr_image):
        def rotate_90():
            return (
                tf.image.rot90(lr_image, k=1),
                tf.image.rot90(hr_image, k=1)
            )
        def rotate_180():
            return (
                tf.image.rot90(lr_image, k=2),
                tf.image.rot90(hr_image, k=2)
            )
        def rotate_270():
            return (
                tf.image.rot90(lr_image, k=3),
                tf.image.rot90(hr_image, k=3)
            )
        rotate_case = tf.random.uniform([1], 0, 4, dtype=tf.int32)
        lr_image, hr_image = tf.case(
            [(tf.equal(rotate_case, 0), rotate_90),
            (tf.equal(rotate_case, 1), rotate_180),
            (tf.equal(rotate_case, 2), rotate_270)],
            default=lambda: (lr_image, hr_image)
        )
    
    def normalize(self, lr_image, hr_image):
        return lr_image / 255, hr_image / 255
    
    def parse_tfrecord(self, tfrecord_file):
        features = {
            'image/img_name': tf.io.FixedLenFeature([], tf.string),
            'image/hr_image': tf.io.FixedLenFeature([], tf.string),
            'image/lr_image': tf.io.FixedLenFeature([], tf.string)
        }
        x = tf.io.parse_single_example(tfrecord, features)
        lr_image = tf.image.decode_png(x['image/lr_image'], channels=3)
        hr_image = tf.image.decode_png(x['image/hr_image'], channels=3)
        lr_image, hr_image = self.apply_random_crop(lr_image, hr_image)
        if self.apply_flip:
            lr_image, hr_image = self.apply_flip_to_images(lr_image, hr_image)
        if self.apply_rotation:
            lr_image, hr_image = self.apply_rotation_to_images(lr_image, hr_image)
        lr_image, hr_image = self.normalize(lr_image, hr_image)
        return lr_image, hr_image
    
    def get_dataset(self, tfrecord_file, batch_size, buffer_size):
        dataset = tf.data.TFRecordDataset(tfrecord_file)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.map(
            self.parse_tfrecord(tfrecord_file),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )
        return dataset



class SRDataset:

    def __init__(self, images, image_size, downsample_scale):
        self.images = images
        self.image_size = image_size
        self.downsample_scale = downsample_scale
    
    def load_image(self, image_file):
        image = tf.io.read_file(image_file)
        image = tf.io.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        return image
    
    def random_crop(self, image):
        return tf.image.random_crop(image, [self.image_size, self.image_size, 3])
    
    def apply_flip_to_images(self, lr_image, hr_image):
        def _flip():
            return (
                tf.image.flip_left_right(lr_image),
                tf.image.flip_left_right(hr_image)
            )
        flip_case = tf.random.uniform([1], 0, 2, dtype=tf.int32)
        lr_image, hr_image = tf.case(
            [(tf.equal(flip_case, 0), _flip)],
            default=lambda: (lr_image, hr_image)
        )
        return lr_image, hr_image
    
    def get_pair(self, image):
        lr_image = tf.image.resize(
            image,
            [self.image_size // self.downsample_scale,] * 2,
            method='bicubic'
        )
        hr_image = image
        lr_image, hr_image = self.apply_flip_to_images(lr_image, hr_image)
        return lr_image, hr_image
    
    def normalize(self, lr_image, hr_image):
        hr_image = hr_image * 2.0 - 1.0
        return lr_image, hr_image
    
    @staticmethod
    def denormalize(image):
        return (image + 1.0) / 2.0
    
    def get_dataset(self, batch_size, buffer_size):
        dataset = tf.data.Dataset.from_tensor_slices(self.images)
        dataset = dataset.map(
            self.load_image,
            num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.map(
            self.random_crop,
            num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.map(
            self.get_pair,
            num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.map(
            self.normalize,
            num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.shuffle(buffer_size).batch(
            batch_size, drop_remainder=True
        ).prefetch(AUTOTUNE)
        return dataset
    
    @staticmethod
    def visualize_batch(dataset, model=None):
        x_batch, y_batch = next(iter(dataset))
        x_batch = x_batch.numpy()
        y_batch = y_batch.numpy()
        c = 0
        if model is None:
            fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 16))
            plt.setp(axes.flat, xticks = [], yticks = [])
            for i, ax in enumerate(axes.flat):
                if i % 2 == 0:
                    ax.imshow(x_batch[c])
                    ax.set_xlabel('Low_Res_' + str(c + 1))
                elif i % 2 == 1:
                    ax.imshow(SRDataset.denormalize(y_batch[c]))
                    ax.set_xlabel('High_Res_' + str(c + 1))
                    c += 1
        else:
            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (16, 16))
            plt.setp(axes.flat, xticks = [], yticks = [])
            for i, ax in enumerate(axes.flat):
                if i % 3 == 0:
                    ax.imshow(x_batch[c])
                    ax.set_xlabel('Low_Res_' + str(c + 1))
                elif i % 3 == 1:
                    ax.imshow(SRDataset.denormalize(y_batch[c]))
                    ax.set_xlabel('High_Res_' + str(c + 1))
                elif i % 3 == 2:
                    ax.imshow(np.squeeze(SRDataset.denormalize(model(np.expand_dims(x_batch[c], axis=0)))))
                    ax.set_xlabel('High_Res_' + str(c + 1))
                    c += 1
        plt.show()