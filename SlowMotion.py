from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import Model


class SlowMotion:
    def __init__(self, checkpoint_file, image_width=128, image_height=128):
        self.image_width = image_width
        self.image_height = image_height
        self.model = Model.Model()
        self.prediction, self.sess = self.model.get(checkpoint_file)

    def gen_frame(self, image_file_start, image_file_end, output_file):
        batch_data_frame1 = np.array([self.model.read_image(image_file_start)])
        batch_data_frame3 = np.array([self.model.read_image(image_file_end)])
        feed_dict = {self.model.input_placeholder: np.concatenate((batch_data_frame1, batch_data_frame3), 3)}
        prediction_np = self.sess.run([self.prediction], feed_dict=feed_dict)
        im = sp.misc.toimage(prediction_np[0][0], cmin=-1.0, cmax=1.0)
        im.save(output_file)

    def gen_frame_im(self, image_file_start_im, image_file_end_im):
        batch_data_frame1 = np.array([image_file_start_im])
        batch_data_frame3 = np.array([image_file_end_im])
        feed_dict = {self.model.input_placeholder: np.concatenate((batch_data_frame1, batch_data_frame3), 3)}
        prediction_np = self.sess.run([self.prediction], feed_dict=feed_dict)
        return sp.misc.toimage(prediction_np[0][0], cmin=-1.0, cmax=1.0)
