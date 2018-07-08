import glob
import moviepy.editor as mpy
import cv2
import shutil
import os
import VideoUtils
import numpy as np
from PIL import Image

#force cpu, if your into that
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class MakeGif:
    def __init__(self, tmp_folder):
        self.tmp_folder = tmp_folder
        self.videoUtils = VideoUtils.VideoUtils(self.tmp_folder, 'models/model')

    def padded(self, current_frame):
        current_frame = current_frame
        index = '' + str(current_frame)
        if current_frame < 10:            index = '0000000' + str(current_frame)
        elif current_frame < 100:         index = '000000' + str(current_frame)
        elif current_frame < 1000:        index = '00000' + str(current_frame)
        elif current_frame < 10000:       index = '0000' + str(current_frame)
        elif current_frame < 100000:      index = '000' + str(current_frame)
        elif current_frame < 1000000:     index = '00' + str(current_frame)
        elif current_frame < 10000000:    index = '0' + str(current_frame)
        return index

    def render(self, input_video, output_gif):
        self.frames_folder = self.videoUtils.create_temp_folder(self.tmp_folder + 'frames/')
        self.videoUtils.extract_frames(input_video, self.frames_folder)
        temp_slow_frames_folder = self.videoUtils.create_temp_folder(self.tmp_folder + 'slow/')
        temp_join_frames_folder = self.videoUtils.create_temp_folder(self.tmp_folder + 'join/')
        last_frame_file = ''
        count_frames = 1
        count_frames_gif = 1
        gen_frame_count=1
        files = glob.glob(self.frames_folder+"*.png")
        files.sort()
        for file in files:
            if count_frames > 1:
                shutil.copyfile(last_frame_file, temp_slow_frames_folder + 'gen_frame_' + self.padded(gen_frame_count) + '.png')
                slow_frame_im = self.videoUtils.get_slow_frame(last_frame_file, file)

                slow_frame_file = temp_slow_frames_folder + 'gen_frame_' + self.padded(gen_frame_count + 1) + '.png'
                cv2.imwrite(slow_frame_file, slow_frame_im)
                video_w, video_h = tuple(slow_frame_im.shape[1::-1])

                shutil.copyfile(file, temp_slow_frames_folder + 'gen_frame_' + self.padded(gen_frame_count + 2) + '.png')

                file_1_im = cv2.imread(last_frame_file)
                file_1_im = cv2.resize(file_1_im, (320,180))

                file_2_im = cv2.resize(slow_frame_im, (320, 180))

                file_3_im = cv2.imread(file)
                file_3_im = cv2.resize(file_3_im, (320, 180))

                video_w = 320
                video_h = 180

                wide_image = np.zeros((video_h, video_w * 2, 3), np.uint8)
                wide_image[0:video_h, 0:video_w] = file_1_im
                wide_image[0:video_h, video_w:video_w * 2] = file_1_im
                cv2.imwrite(temp_join_frames_folder + 'gen_frame_' + self.padded(count_frames_gif) + '.png', wide_image)
                image = Image.open(temp_join_frames_folder + 'gen_frame_' + self.padded(count_frames_gif) + '.png')
                image.save(temp_join_frames_folder + 'gen_frame_' + self.padded(count_frames_gif) + '.png', quality=20,optimize=True)
                count_frames_gif += 1

                wide_image = np.zeros((video_h, video_w * 2, 3), np.uint8)
                wide_image[0:video_h, 0:video_w] = file_1_im
                wide_image[0:video_h, video_w:video_w * 2] = file_2_im
                cv2.imwrite(temp_join_frames_folder + 'gen_frame_' + self.padded(count_frames_gif) + '.png', wide_image)
                image = Image.open(temp_join_frames_folder + 'gen_frame_' + self.padded(count_frames_gif) + '.png')
                image.save(temp_join_frames_folder + 'gen_frame_' + self.padded(count_frames_gif) + '.png', quality=20,optimize=True)
                count_frames_gif += 1

                # wide_image = np.zeros((video_h, video_w * 2, 3), np.uint8)
                # wide_image[0:video_h, 0:video_w] = file_1_im
                # wide_image[0:video_h, video_w:video_w * 2] = file_3_im
                # cv2.imwrite(temp_join_frames_folder + 'gen_frame_' + self.padded(count_frames_gif) + '.png', wide_image)
                # image = Image.open(temp_join_frames_folder + 'gen_frame_' + self.padded(count_frames_gif) + '.png')
                # image.save(temp_join_frames_folder + 'gen_frame_' + self.padded(count_frames_gif) + '.png', quality=20, optimize=True)
                # count_frames_gif += 1

                gen_frame_count += 3
            last_frame_file = file
            count_frames += 1

        file_list = glob.glob(self.tmp_folder + 'join/'+'*.png')
        file_list.sort()
        # file_list = file_list[20:40]
        clip = mpy.ImageSequenceClip(file_list, fps=15)
        clip.write_gif('{}.gif'.format(output_gif), fps=15)

        # self.videoUtils.delete_temp_folder(self.frames_folder)
        # self.videoUtils.delete_temp_folder(temp_slow_frames_folder)
        # self.videoUtils.delete_temp_folder(temp_join_frames_folder)




makeGif = MakeGif('tmp/')
makeGif.render('/media/administrator/E2C8EAECC8EABDC3/tensorflow/slowmotionpro/videos/UCF-101/Fencing/v_Fencing_g04_c01.avi','output')


# video_file
# cap = cv2.VideoCapture(video_file)
# current_frame = 1
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         output_image_file = output_folder + 'frame_' + self.padded(current_frame) + '.png'
#         cv2.imwrite(output_image_file, frame)
#         current_frame += 1
#     else:
#         break
#
# gif_name = 'outputName'
# fps = 12
# file_list = glob.glob('*.png') # Get all the pngs in the current directory
# list.sort(file_list, key=lambda x: int(x.split('_')[1].split('.png')[0])) # Sort the images by #, this may need to be tweaked for your use case
# clip = mpy.ImageSequenceClip(file_list, fps=fps)
# clip.write_gif('{}.gif'.format(gif_name), fps=fps)