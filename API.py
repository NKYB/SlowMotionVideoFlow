from flask import Flask, request
import VideoUtils
import urllib2

app = Flask(__name__) #create the Flask app

temp_folder = '/media/administrator/E2C8EAECC8EABDC3/tmp/'
checkpoint_file = '/media/administrator/E2C8EAECC8EABDC3/tensorflow/slowmotionpro/voxel-flow/voxel_flow_checkpoints/model.ckpt-360000'

@app.route('/')
def query_example():
    video_url = request.args.get('url')

    videoUtils = VideoUtils.VideoUtils(temp_folder, checkpoint_file)
    temp_video_file = temp_folder + videoUtils.rnd_folder_label() + '.avi'

    with open(temp_video_file, 'wb') as f:
        f.write(urllib2.urlopen(video_url).read())
        f.close()

    output_video_file = temp_folder + 'output_' + videoUtils.rnd_folder_label() + '.avi'
    videoUtils.slow_down_video(temp_video_file, output_video_file)

    return 'Url: ' + video_url + ' Output Video File: ' + output_video_file


if __name__ == '__main__':
    app.run(debug=True, port=5000) #run app in debug mode on port 5000
