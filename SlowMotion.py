from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy as sp

class SlowMotion:
    def __init__(self, checkpoint_file, image_width=128, image_height=128):
        self.image_width = image_width
        self.image_height = image_height
        with tf.Graph().as_default():

            self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.image_height, self.image_width, 6))

            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0001)):
                # Define network
                batch_norm_params = {
                    'decay': 0.9997,
                    'epsilon': 0.001,
                    'is_training': False,
                }
                with slim.arg_scope([slim.batch_norm], is_training=False, updates_collections=None):
                    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params):
                        net = slim.conv2d(self.input_placeholder, 64, [5, 5], stride=1, scope='conv1')
                        net = slim.max_pool2d(net, [2, 2], scope='pool1')
                        net = slim.conv2d(net, 128, [5, 5], stride=1, scope='conv2')
                        net = slim.max_pool2d(net, [2, 2], scope='pool2')
                        net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3')
                        net = slim.max_pool2d(net, [2, 2], scope='pool3')
                        net = tf.image.resize_bilinear(net, [64, 64])
                        net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv4')
                        net = tf.image.resize_bilinear(net, [128, 128])
                        net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv5')
                        net = tf.image.resize_bilinear(net, [256, 256])
                        net = slim.conv2d(net, 64, [5, 5], stride=1, scope='conv6')
            net = slim.conv2d(net, 3, [5, 5], stride=1, activation_fn=tf.tanh,normalizer_fn=None, scope='conv7')

            flow = net[:, :, :, 0:2]
            mask = tf.expand_dims(net[:, :, :, 2], 3)

            grid_x, grid_y = self.meshgrid(self.image_height, self.image_width)
            grid_x = tf.tile(grid_x, [1, 1, 1])
            grid_y = tf.tile(grid_y, [1, 1, 1])

            flow = 0.5 * flow

            coor_x_1 = grid_x + flow[:, :, :, 0]
            coor_y_1 = grid_y + flow[:, :, :, 1]

            coor_x_2 = grid_x - flow[:, :, :, 0]
            coor_y_2 = grid_y - flow[:, :, :, 1]

            output_1 = self.bilinear_interp(self.input_placeholder[:, :, :, 0:3], coor_x_1, coor_y_1, 'interpolate')
            output_2 = self.bilinear_interp(self.input_placeholder[:, :, :, 3:6], coor_x_2, coor_y_2, 'interpolate')

            mask = 0.5 * (1.0 + mask)
            mask = tf.tile(mask, [1, 1, 1, 3])
            self.model = tf.multiply(mask, output_1) + tf.multiply(1.0 - mask, output_2)

            self.sess = tf.Session()
            restorer = tf.train.Saver()
            restorer.restore(self.sess, checkpoint_file)

    def bilinear_interp(self, im, x, y, name):
        with tf.variable_scope(name):
            x = tf.reshape(x, [-1])
            y = tf.reshape(y, [-1])

            # constants
            num_batch = tf.shape(im)[0]
            _, height, width, channels = im.get_shape().as_list()

            x = tf.to_float(x)
            y = tf.to_float(y)

            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            zero = tf.constant(0, dtype=tf.int32)

            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            x = (x + 1.0) * (width_f - 1.0) / 2.0
            y = (y + 1.0) * (height_f - 1.0) / 2.0

            # Sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)

            dim2 = width
            dim1 = width * height

            # Create base index
            base = tf.range(num_batch) * dim1
            base = tf.reshape(base, [-1, 1])
            base = tf.tile(base, [1, height * width])
            base = tf.reshape(base, [-1])

            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # Use indices to look up pixels
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.to_float(im_flat)
            pixel_a = tf.gather(im_flat, idx_a)
            pixel_b = tf.gather(im_flat, idx_b)
            pixel_c = tf.gather(im_flat, idx_c)
            pixel_d = tf.gather(im_flat, idx_d)

            # Interpolate the values
            x1_f = tf.to_float(x1)
            y1_f = tf.to_float(y1)

            wa = tf.expand_dims((x1_f - x) * (y1_f - y), 1)
            wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
            wc = tf.expand_dims((1.0 - (x1_f - x)) * (y1_f - y), 1)
            wd = tf.expand_dims((1.0 - (x1_f - x)) * (1.0 - (y1_f - y)), 1)

            output = tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
            output = tf.reshape(output, shape=tf.stack([num_batch, height, width, channels]))
            return output

    def meshgrid(self, height, width):
        with tf.variable_scope('meshgrid'):
            x_t = tf.matmul(
                tf.ones(shape=tf.stack([height, 1])),
                tf.transpose(
                    tf.expand_dims(
                        tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(
                tf.expand_dims(
                    tf.linspace(-1.0, 1.0, height), 1),
                tf.ones(shape=tf.stack([1, width])))
            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            # grid_x = tf.reshape(x_t_flat, [1, height, width, 1])
            # grid_y = tf.reshape(y_t_flat, [1, height, width, 1])
            grid_x = tf.reshape(x_t_flat, [1, height, width])
            grid_y = tf.reshape(y_t_flat, [1, height, width])
            return grid_x, grid_y

    def read_image(self, image_file):
        im = sp.misc.imread(image_file)
        im = im / 127.5 - 1.0
        return im

    def gen_frame(self, image_file_start, image_file_end, output_file):
        batch_data_frame1 = np.array([self.read_image(image_file_start)])
        batch_data_frame3 = np.array([self.read_image(image_file_end)])
        feed_dict = {self.input_placeholder: np.concatenate((batch_data_frame1, batch_data_frame3), 3)}
        prediction_np = self.sess.run([self.model], feed_dict=feed_dict)
        im = sp.misc.toimage(prediction_np[0][0], cmin=-1.0, cmax=1.0)
        im.save(output_file)

    def gen_frame_im(self, image_file_start_im, image_file_end_im):
        batch_data_frame1 = np.array([image_file_start_im])
        batch_data_frame3 = np.array([image_file_end_im])
        feed_dict = {self.input_placeholder: np.concatenate((batch_data_frame1, batch_data_frame3), 3)}
        prediction_np = self.sess.run([self.model], feed_dict=feed_dict)
        return sp.misc.toimage(prediction_np[0][0], cmin=-1.0, cmax=1.0)

# DO YOU TEST?
# slowMotion = SlowMotion('/media/administrator/E2C8EAECC8EABDC3/tensorflow/slowmotionpro/voxel-flow/voxel_flow_checkpoints/model.ckpt-360000')
# for x in range(1,10):
#     slowMotion.gen_frame(
#         '/media/administrator/E2C8EAECC8EABDC3/tensorflow/slowmotionpro/videos/frames/v_ApplyEyeMakeup_g01_c01_frame_' + str(x) + '.png',
#         '/media/administrator/E2C8EAECC8EABDC3/tensorflow/slowmotionpro/videos/frames/v_ApplyEyeMakeup_g01_c01_frame_' + str(x + 2) + '.png',
#         'test_' + str(x + 1) + '.png'
#     )