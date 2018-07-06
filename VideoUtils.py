import cv2
import os
import random
import string
import shutil
import glob
import SlowMotion

class VideoUtils:
    def __init__(self, tmp_folder, checkpoint_file="", image_width=128, image_height=128):
        self.tmp_folder = tmp_folder
        self.image_width = image_width
        self.image_height = image_height
        if checkpoint_file != "":
            self.slowMotion = SlowMotion.SlowMotion(checkpoint_file)


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

    def rnd_folder_label(self):
        return ''.join([random.choice(string.ascii_letters + string.digits) for n in range(16)])

    def extract_frames(self, video_file, output_folder):

        cap = cv2.VideoCapture(video_file)
        current_frame = 1
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                output_image_file = output_folder + 'frame_' + self.padded(current_frame) + '.png'
                cv2.imwrite(output_image_file, frame)
                current_frame += 1
            else:
                break

    def create_temp_folder(self, path):
        self.delete_temp_folder(path)
        os.makedirs(path)
        return path;

    def delete_temp_folder(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)

    def sliding_window(self, image, stepSize=200, windowSize=(256,256)):
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                window = image[y:y + windowSize[1], x:x + windowSize[0]]
                h,w,d = window.shape
                yield (x, y, h, w, window)

    def get_image_parts(self, input_file):
        color = [0, 0, 0]
        im_input = cv2.imread(input_file)
        image_parts = []
        for (x, y, h, w, window) in self.sliding_window(im_input, 200, (self.image_height, self.image_width)):
            if h < self.image_height or w < self.image_width:
                border_im = cv2.copyMakeBorder(window, 0, self.image_height-h, 0, self.image_width-w, cv2.BORDER_CONSTANT,value=color)
                image_parts.append({'x': x, 'y': y, 'w': w, 'h': h, 'im': border_im})
            else:
                image_parts.append({'x':x,'y':y,'w':w,'h':h,'im':window})
        return image_parts

    def get_slow_frame(self, image_file_start, image_file_end):
        im_input = cv2.imread(image_file_start)
        image_parts_start = self.get_image_parts(image_file_start)
        image_parts_end = self.get_image_parts(image_file_end)
        image_parts_slow = []
        for key, value in enumerate(image_parts_start):
            image_part_start = image_parts_start[key]
            image_part_end = image_parts_end[key]
            image_part_slow = image_part_start.copy()

            scratch_file = self.tmp_folder + self.rnd_folder_label() + '.png'
            im_start = image_part_start['im']
            cv2.imwrite(scratch_file, im_start)
            im_start = self.slowMotion.model.read_image(scratch_file)

            im_end = image_part_end['im']
            cv2.imwrite(scratch_file, im_end)
            im_end = self.slowMotion.model.read_image(scratch_file)

            im = self.slowMotion.gen_frame_im(im_start, im_end)
            im.save(scratch_file)
            im = cv2.imread(scratch_file)
            os.remove(scratch_file)

            image_part_slow['im'] = im
            image_parts_slow.append(image_part_slow)
        for key, image_part_slow in enumerate(image_parts_slow):
            im_crop = image_part_slow['im']
            im_crop = im_crop[0:image_part_slow['h'], 0:image_part_slow['w']]
            im_input[image_part_slow['y']:image_part_slow['y']+image_part_slow['h'], image_part_slow['x']:image_part_slow['x']+image_part_slow['w']] = im_crop
        return im_input

    def slow_down_video(self, video_file, output_video_file):
        video_h = self.image_height
        video_w = self.image_width
        temp_frames_folder = self.create_temp_folder(self.tmp_folder + self.rnd_folder_label() + '/')
        temp_slow_frames_folder = self.create_temp_folder(self.tmp_folder + self.rnd_folder_label() + '/')

        self.extract_frames(video_file, temp_frames_folder)

        # slow down frames
        last_frame_file = ''
        count_frames = 1
        gen_frame_count=1
        files = glob.glob(temp_frames_folder+"*.png")
        files.sort()
        for file in files:
            if count_frames > 1:
                shutil.copyfile(last_frame_file, temp_slow_frames_folder + 'gen_frame_' + self.padded(gen_frame_count) + '.png')
                slow_frame_im = self.get_slow_frame(last_frame_file, file)
                cv2.imwrite(temp_slow_frames_folder + 'gen_frame_' + self.padded(gen_frame_count + 1) + '.png', slow_frame_im)
                video_w, video_h = tuple(slow_frame_im.shape[1::-1])
                shutil.copyfile(file, temp_slow_frames_folder + 'gen_frame_' + self.padded(gen_frame_count + 2) + '.png')
                gen_frame_count += 3
            last_frame_file = file
            count_frames += 1

        # create video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(output_video_file, fourcc, 20.0, (video_w, video_h))
        images = glob.glob(temp_slow_frames_folder + "*.png")
        images.sort()
        for image in images:
            video.write(cv2.imread(os.path.join(temp_slow_frames_folder, image)))
        cv2.destroyAllWindows()
        video.release()

        self.delete_temp_folder(temp_frames_folder)
        self.delete_temp_folder(temp_slow_frames_folder)

# DO YOU TEST
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
videoUtils = VideoUtils('tmp/', 'models/model')
videoUtils.slow_down_video('videos/fencing/v_Fencing_g01_c01.avi', 'output_v_Fencing_g01_c01.avi')