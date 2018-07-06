from flask import Flask, request
import VideoUtils
import urllib
import shutil
import os

app = Flask(__name__) #create the Flask app

temp_folder = 'tmp/'
if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
os.makedirs(temp_folder)

checkpoint_file = 'models/model'

#force cpu, if your into that
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

@app.route('/')
def get_url():
    video_url = request.args.get('url')

    videoUtils = VideoUtils.VideoUtils(temp_folder, checkpoint_file)
    temp_video_file = temp_folder + videoUtils.rnd_folder_label() + '.avi'

    with open(temp_video_file, 'wb') as f:
        f.write(urllib.request.urlopen(video_url).read())
        f.close()

    output_video_file = temp_folder + 'output_' + videoUtils.rnd_folder_label() + '.avi'
    videoUtils.slow_down_video(temp_video_file, output_video_file)

    return 'Url: ' + video_url + ' Output Video File: ' + output_video_file


if __name__ == '__main__':
    app.run(debug=True, port=5000) #run app in debug mode on port 5000

#DO YOU TEST
# http://127.0.0.1:5000/?url=http://www.engr.colostate.edu/me/facil/dynamics/files/drop.avi
