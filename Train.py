from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy as sp
import VideoUtils
import cv2
import random


class Train:
    def __init__(self, tmp_folder, model_output_folder, input_video_folder, output_images_folder, video_extension="avi", image_width=128, image_height=128, learning_rate=0.0003, steps=1100000):
        self.tmp_folder = tmp_folder
        self.model_output_folder = model_output_folder
        self.input_video_folder = input_video_folder
        self.output_images_folder = output_images_folder
        self.video_extension = video_extension
        self.image_width = image_width
        self.image_height = image_height
        self.learning_rate = learning_rate
        self.steps = steps
        self.output_video_frames_list = self.model_output_folder + 'output_video_frames_list.csv'

    def l1_loss(self, predictions, targets):
        """Implements tensorflow l1 loss.
        Args:
        Returns:
        """
        total_elements = (tf.shape(targets)[0] * tf.shape(targets)[1] * tf.shape(targets)[2]
                          * tf.shape(targets)[3])
        total_elements = tf.to_float(total_elements)

        loss = tf.reduce_sum(tf.abs(predictions - targets))
        loss = tf.div(loss, total_elements)
        return loss

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

    def train(self):
        # read file list
        #if os.path.exists(self.output_video_frames_list):
        #    os.remove(self.output_video_frames_list)
        if not os.path.exists(self.output_video_frames_list):
            print('Create training file, this could take a while :)' + self.output_video_frames_list)
            exit()

            videoUtils = VideoUtils.VideoUtils(self.tmp_folder, '', self.image_width, self.image_height)
            videoUtils.create_temp_folder(self.output_images_folder)
            for folder in glob.glob(self.input_video_folder + "*/"):
                folder_name = folder.replace(self.input_video_folder, '').replace('/', '')
                for file in glob.glob(folder + "*."+self.video_extension):
                    file_name = file.replace(self.input_video_folder + folder_name + '/', '').replace('.'+self.video_extension, '')
                    print(folder,folder_name,file,file_name)

                    new_video_folder = self.output_images_folder + folder_name + '/' + file_name + '/'
                    videoUtils.create_temp_folder(new_video_folder)

                    f1 = open(self.output_video_frames_list, 'a')
                    cap = cv2.VideoCapture(file)
                    current_frame = 1
                    while (cap.isOpened()):
                        ret, frame = cap.read()
                        if ret == True:
                            output_image_file = new_video_folder + 'frame_' + videoUtils.padded(current_frame) + '.png'
                            cv2.imwrite(output_image_file, frame)
                            image_parts = videoUtils.get_image_parts(output_image_file)
                            count_keys = 1
                            for key, value in enumerate(image_parts):
                                output_image_file_part = output_image_file.replace('.png','_'+str(key+1)+'.png')
                                cv2.imwrite(output_image_file_part, value['im'])
                                count_keys += 1

                            if current_frame > 2:
                                for x in range(1,count_keys):
                                    line = ''
                                    line += new_video_folder + 'frame_' + videoUtils.padded(current_frame-2) + '_'+str(x)+'.png,'
                                    line += new_video_folder + 'frame_' + videoUtils.padded(current_frame-1) + '_' + str(x) + '.png,'
                                    line += new_video_folder + 'frame_' + videoUtils.padded(current_frame)   + '_' + str(x) + '.png\n'
                                    f1.write(line)
                            current_frame += 1
                            os.remove(output_image_file)
                        else:
                            break

                    f1.close()
                    # exit()
                    #break
                #break

            #f1.close()


        with tf.Graph().as_default():
            input_placeholder = tf.placeholder(tf.float32, shape=(None, self.image_height, self.image_width, 6))
            target_placeholder = tf.placeholder(tf.float32, shape=(None, self.image_height, self.image_width, 3))
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
                        net = slim.conv2d(input_placeholder, 64, [5, 5], stride=1, scope='conv1')
                        net = slim.max_pool2d(net, [2, 2], scope='pool1')
                        net = slim.conv2d(net, 128, [5, 5], stride=1, scope='conv2')
                        net = slim.max_pool2d(net, [2, 2], scope='pool2')
                        net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3')
                        net = slim.max_pool2d(net, [2, 2], scope='pool3')
                        net = tf.image.resize_bilinear(net, [64, 64])
                        net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv4')
                        net = tf.image.resize_bilinear(net, [128, 128])
                        net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv5')
                        net = tf.image.resize_bilinear(net, [self.image_height, self.image_width])
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

            output_1 = self.bilinear_interp(input_placeholder[:, :, :, 0:3], coor_x_1, coor_y_1, 'interpolate')
            output_2 = self.bilinear_interp(input_placeholder[:, :, :, 3:6], coor_x_2, coor_y_2, 'interpolate')

            mask = 0.5 * (1.0 + mask)
            mask = tf.tile(mask, [1, 1, 1, 3])
            prediction = tf.multiply(mask, output_1) + tf.multiply(1.0 - mask, output_2)
            reproduction_loss = self.l1_loss(prediction, target_placeholder)
            total_loss = reproduction_loss

            opt = tf.train.AdamOptimizer(self.learning_rate)
            grads = opt.compute_gradients(total_loss)
            update_op = opt.apply_gradients(grads)

            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            summaries.append(tf.summary.scalar('total_loss', total_loss))
            summaries.append(tf.summary.scalar('reproduction_loss', reproduction_loss))
            summaries.append(tf.summary.image('Input Image', input_placeholder, 3))
            summaries.append(tf.summary.image('Output Image', prediction, 3))
            summaries.append(tf.summary.image('Target Image', target_placeholder, 3))

            saver = tf.train.Saver(tf.all_variables())
            summary_op = tf.summary.merge_all()
            init = tf.initialize_all_variables()
            sess = tf.Session()
            sess.run(init)
            summary_writer = tf.summary.FileWriter(self.model_output_folder,graph=sess.graph)

            offset = os.path.getsize(self.output_video_frames_list) - 1500
            offset -= 582431524
            print(offset)
            # exit()
            f = open(self.output_video_frames_list)


            for step in xrange(0, self.steps):
                f.seek(offset)
                f.readline()
                list_of_image_set = f.readline()
                image_files = list_of_image_set.split(",")
                image_file_1 = image_files[0].rstrip()
                image_file_2 = image_files[1].rstrip()
                image_file_3 = image_files[2].rstrip()
                image_raw_1 = [self.read_image(image_file_1)]
                image_raw_2 = [self.read_image(image_file_2)]
                image_raw_3 = [self.read_image(image_file_3)]

                feed_dict = {input_placeholder: np.concatenate((image_raw_1, image_raw_3), 3),target_placeholder: image_raw_2}
                _, loss_value = sess.run([update_op, total_loss], feed_dict=feed_dict)

                if step % 10 == 0:
                    print("Loss at step %d: %f" % (step, loss_value))

                if step % 5000 == 0 or (step + 1) == self.steps:
                    checkpoint_path = os.path.join(self.model_output_folder, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)



# DO YOU TEST?
trainer = Train(
    '/tmp/',
    '/mnt/DC7C16307C160644/tensorflow/slowmotionpro/src/models/',
    '/mnt/DC7C16307C160644/tensorflow/slowmotionpro/videos/UCF-101/',
    '/mnt/DC7C16307C160644/tensorflow/slowmotionpro/videos/frames/',
    'avi',
    128,
    128
)
trainer.train()
