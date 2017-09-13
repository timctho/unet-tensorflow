import tensorflow as tf
import tensorflow.contrib as tc


class UNet():
    def __init__(self, input_size, is_training=False):
        self.input_size = input_size
        self.input = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3])

        with tf.variable_scope('UNet/down_full_size'):
            down0a = tc.layers.conv2d(self.input, 32, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                      normalizer_params={'is_training': is_training})
            down0b = tc.layers.conv2d(down0a, 32, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                      normalizer_params={'is_training': is_training})
            down0c = tc.layers.max_pool2d(down0b, (2, 2), padding='same')

        with tf.variable_scope('UNet/down_1-2_size'):
            down1a = tc.layers.conv2d(down0c, 32, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                      normalizer_params={'is_training': is_training})
            down1b = tc.layers.conv2d(down1a, 32, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                      normalizer_params={'is_training': is_training})
            down1c = tc.layers.max_pool2d(down1b, (2, 2), padding='same')

        with tf.variable_scope('UNet/down_1-4_size'):
            down2a = tc.layers.conv2d(down1c, 32, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                      normalizer_params={'is_training': is_training})
            down2b = tc.layers.conv2d(down2a, 32, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                      normalizer_params={'is_training': is_training})
            down2c = tc.layers.max_pool2d(down2b, (2, 2), padding='same')

        with tf.variable_scope('UNet/down_1-8_size'):
            down3a = tc.layers.conv2d(down2c, 64, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                      normalizer_params={'is_training': is_training})
            down3b = tc.layers.conv2d(down3a, 64, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                      normalizer_params={'is_training': is_training})
            down3c = tc.layers.max_pool2d(down3b, (2, 2), padding='same')

        with tf.variable_scope('UNet/down_1-16_size'):
            down4a = tc.layers.conv2d(down3c, 128, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                      normalizer_params={'is_training': is_training})
            down4b = tc.layers.conv2d(down4a, 128, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                      normalizer_params={'is_training': is_training})
            down4c = tc.layers.max_pool2d(down4b, (2, 2), padding='same')

        with tf.variable_scope('UNet/down_1-32_size'):
            down5a = tc.layers.conv2d(down4c, 256, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                      normalizer_params={'is_training': is_training})
            down5b = tc.layers.conv2d(down5a, 256, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                      normalizer_params={'is_training': is_training})
            down5c = tc.layers.max_pool2d(down5b, (2, 2), padding='same')

        # with tf.variable_scope('UNet/down_1-64_size'):
        #     down6a = tc.layers.conv2d(down5c, 512, (3, 3), normalizer_fn=tc.layers.batch_norm,
        #                               normalizer_params={'is_training': is_training})
        #     down6b = tc.layers.conv2d(down6a, 512, (3, 3), normalizer_fn=tc.layers.batch_norm,
        #                               normalizer_params={'is_training': is_training})
        #     down6c = tc.layers.max_pool2d(down6b, (2, 2), padding='same')

        with tf.variable_scope('UNet/down_1-128_size'):
            down7a = tc.layers.conv2d(down5c, 512, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                      normalizer_params={'is_training': is_training})
            down7b = tc.layers.conv2d(down7a, 512, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                      normalizer_params={'is_training': is_training})

        # with tf.variable_scope('UNet/up_1-64_size'):
        #     up6a = tc.layers.conv2d_transpose(down7b, 512, (3, 3), 2, normalizer_fn=tc.layers.batch_norm,
        #                                       normalizer_params={'is_training': is_training})
        #     up6b = tf.concat([up6a, down6b], axis=3)
        #     up6c = tc.layers.conv2d(up6b, 512, (3, 3), normalizer_fn=tc.layers.batch_norm,
        #                             normalizer_params={'is_training': is_training})
        #     up6d = tc.layers.conv2d(up6c, 512, (3, 3), normalizer_fn=tc.layers.batch_norm,
        #                             normalizer_params={'is_training': is_training})
        #     up6e = tc.layers.conv2d(up6d, 512, (3, 3), normalizer_fn=tc.layers.batch_norm,
        #                             normalizer_params={'is_training': is_training})

        with tf.variable_scope('UNet/up_1-32_size'):
            # up5a = tc.layers.conv2d_transpose(down7b, 256, (3, 3), 2, normalizer_fn=tc.layers.batch_norm,
            #                                   normalizer_params={'is_training': is_training})
            shape = down7b.get_shape().as_list()
            up5a = tf.image.resize_bilinear(down7b, size=[2*shape[1], 2*shape[2]], align_corners=True)
            up5b = tf.concat([up5a, down5b], axis=3)
            up5c = tc.layers.conv2d(up5b, 256, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})
            up5d = tc.layers.conv2d(up5c, 256, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})
            up5e = tc.layers.conv2d(up5d, 256, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})

        with tf.variable_scope('UNet/up_1-16_size'):
            # up4a = tc.layers.conv2d_transpose(up5e, 128, (3, 3), 2, normalizer_fn=tc.layers.batch_norm,
            #                                   normalizer_params={'is_training': is_training})
            shape = up5e.get_shape().as_list()
            up4a = tf.image.resize_bilinear(up5e, size=[2*shape[1], 2*shape[2]], align_corners=True)
            up4b = tf.concat([up4a, down4b], axis=3)
            up4c = tc.layers.conv2d(up4b, 128, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})
            up4d = tc.layers.conv2d(up4c, 128, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})
            up4e = tc.layers.conv2d(up4d, 128, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})

        with tf.variable_scope('UNet/up_1-8_size'):
            # up3a = tc.layers.conv2d_transpose(up4e, 64, (3, 3), 2, normalizer_fn=tc.layers.batch_norm,
            #                                   normalizer_params={'is_training': is_training})
            shape = up4e.get_shape().as_list()
            up3a = tf.image.resize_bilinear(up4e, size=[2*shape[1], 2*shape[2]], align_corners=True)
            up3b = tf.concat([up3a, down3b], axis=3)
            up3c = tc.layers.conv2d(up3b, 64, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})
            up3d = tc.layers.conv2d(up3c, 64, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})
            up3e = tc.layers.conv2d(up3d, 64, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})

        with tf.variable_scope('UNet/up_1-4_size'):
            # up2a = tc.layers.conv2d_transpose(up3e, 32, (3, 3), 2, normalizer_fn=tc.layers.batch_norm,
            #                                   normalizer_params={'is_training': is_training})
            shape = up3e.get_shape().as_list()
            up2a = tf.image.resize_bilinear(up3e, size=[2*shape[1], 2*shape[2]], align_corners=True)
            up2b = tf.concat([up2a, down2b], axis=3)
            up2c = tc.layers.conv2d(up2b, 32, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})
            up2d = tc.layers.conv2d(up2c, 32, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})
            up2e = tc.layers.conv2d(up2d, 32, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})

        with tf.variable_scope('UNet/up_1-2_size'):
            # up1a = tc.layers.conv2d_transpose(up2e, 16, (3, 3), 2, normalizer_fn=tc.layers.batch_norm,
            #                                   normalizer_params={'is_training': is_training})
            shape = up2e.get_shape().as_list()
            up1a = tf.image.resize_bilinear(up2e, size=[2*shape[1], 2*shape[2]], align_corners=True)
            up1b = tf.concat([up1a, down1b], axis=3)
            up1c = tc.layers.conv2d(up1b, 32, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})
            up1d = tc.layers.conv2d(up1c, 32, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})
            up1e = tc.layers.conv2d(up1d, 32, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})

        with tf.variable_scope('UNet/up_full_size'):
            # up0a = tc.layers.conv2d_transpose(up1e, 8, (3, 3), 2, normalizer_fn=tc.layers.batch_norm,
            #                                   normalizer_params={'is_training': is_training})
            shape = up1e.get_shape().as_list()
            up0a = tf.image.resize_bilinear(up1e, size=[2*shape[1], 2*shape[2]], align_corners=True)
            up0b = tf.concat([up0a, down0b], axis=3)
            up0c = tc.layers.conv2d(up0b, 32, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})
            up0d = tc.layers.conv2d(up0c, 32, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})
            up0e = tc.layers.conv2d(up0d, 32, (3, 3), normalizer_fn=tc.layers.batch_norm,
                                    normalizer_params={'is_training': is_training})

        with tf.variable_scope('UNet/output_mask'):
            self.output_mask = tc.layers.conv2d(up0e, 1, [1, 1], activation_fn=None)

    def train(self, learning_rate):
        self.gs = tc.framework.get_or_create_global_step()
        self.gt_mask = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 1])
        self.loss_w = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 1])
        self.loss_w_add = self.loss_w + 1
        self.loss_out = self.loss_w_add

        # dice loss
        flat_gt_mask = tc.layers.flatten(self.gt_mask)
        flat_output_mask = tc.layers.flatten(self.output_mask)
        flat_output_mask = tf.nn.sigmoid(flat_output_mask)
        intersection = tf.multiply(flat_gt_mask, flat_output_mask)
        self.dice_acc = (2. * tf.reduce_sum(intersection) + 1.) / (
            tf.reduce_sum(flat_gt_mask) + tf.reduce_sum(flat_output_mask) + 1.)
        tf.summary.scalar('dice acc', self.dice_acc)

        # cross entropy loss
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.gt_mask,
                                                            logits=self.output_mask)

        # reinforce on edge
        self.loss = tf.multiply(self.loss, self.loss_w_add)

        self.loss = tf.reduce_mean(self.loss)
        tf.summary.scalar('wce loss', self.loss)

        self.loss += (1 - self.dice_acc)
        tf.summary.scalar('total loss', self.loss)

        self.lr = tf.train.exponential_decay(learning_rate,
                                             global_step=self.gs,
                                             decay_rate=0.5,
                                             decay_steps=50000)
        tf.summary.scalar('learning rate', self.lr)

        self.train_op = tc.layers.optimize_loss(loss=self.loss,
                                                global_step=self.gs,
                                                learning_rate=self.lr,
                                                optimizer='RMSProp')
        self.merged_summary = tf.summary.merge_all()
        return self.train_op

    @property
    def vars(self):
        return [i for i in tf.global_variables_initializer() if 'UNet' in i.name]
