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
import Model
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



    def train(self):
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
                    while cap.isOpened():
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

        model = Model.Model()
        prediction = model.get()

        reproduction_loss = model.l1_loss(prediction, model.target_placeholder)
        total_loss = reproduction_loss

        opt = tf.train.AdamOptimizer(self.learning_rate)
        grads = opt.compute_gradients(total_loss)
        update_op = opt.apply_gradients(grads)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar('total_loss', total_loss))
        summaries.append(tf.summary.scalar('reproduction_loss', reproduction_loss))
        summaries.append(tf.summary.image('Input Image', model.input_placeholder, 3))
        summaries.append(tf.summary.image('Output Image', prediction, 3))
        summaries.append(tf.summary.image('Target Image', model.target_placeholder, 3))

        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        summary_writer = tf.summary.FileWriter(self.model_output_folder,graph=sess.graph)

        offset = os.path.getsize(self.output_video_frames_list) - 1500
        offset -= 582431524
        f = open(self.output_video_frames_list)

        for step in range(0, self.steps):
            f.seek(offset)
            f.readline()
            list_of_image_set = f.readline()
            image_files = list_of_image_set.split(",")
            image_file_1 = image_files[0].rstrip()
            image_file_2 = image_files[1].rstrip()
            image_file_3 = image_files[2].rstrip()
            image_raw_1 = [model.read_image(image_file_1)]
            image_raw_2 = [model.read_image(image_file_2)]
            image_raw_3 = [model.read_image(image_file_3)]

            feed_dict = {model.input_placeholder: np.concatenate((image_raw_1, image_raw_3), 3), model.target_placeholder: image_raw_2}
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
