# SlowMotionVideoFlow
High quality slow motion video generation using deep neural nets and optical flow.

Supports many video types through OpenCV and video sizes.

<img src='./images/demo.gif' width=640>

## Requirements
- Tensorflow
- OpenCV

## Quickstart

If you just want to slow motion some video. (Model is in the models folder)
```
videoUtils = VideoUtils('tmp/', 'models/model')
videoUtils.slow_down_video('videos/fencing/v_Fencing_g01_c02.avi', 'output_v_Fencing_g01_c02.avi')
```

If you want to train a new model
```
trainer = Train(
    'tmp/',
    'models/',
    'videos/',
    'avi',
    128,
    128
)
trainer.train()
```

## Training Videos
Info on videos - (http://crcv.ucf.edu/data/UCF101.php)

Download Videos (http://crcv.ucf.edu/data/UCF101/UCF101.rar)

## Research Source
`Deep Voxel Flow (DVF)` is the author's re-implementation of the video frame synthesizer described in:
"Video Frame Synthesis using Deep Voxel Flow"
[Ziwei Liu](https://liuziwei7.github.io/), [Raymond A. Yeh](http://www.isle.illinois.edu/~yeh17/), [Xiaoou Tang](http://www.ie.cuhk.edu.hk/people/xotang.shtml), [Yiming Liu](http://bitstream9.me/), [Aseem Agarwala](http://www.agarwala.org/) (CUHK & UIUC & Google Research)
in International Conference on Computer Vision (ICCV) 2017, Oral Presentation

Further information please contact [Ziwei Liu](https://liuziwei7.github.io/).

```
@inproceedings{liu2017voxelflow,
 author = {Ziwei Liu, Raymond Yeh, Xiaoou Tang, Yiming Liu, and Aseem Agarwala},
 title = {Video Frame Synthesis using Deep Voxel Flow},
 booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
 month = {October},
 year = {2017}
}
```

## Forked from this project
https://github.com/liuziwei7/voxel-flow


