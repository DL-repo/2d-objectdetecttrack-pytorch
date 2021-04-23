# PyTorch Object Detection and Tracking

Object detection in images, and tracking across video frames.

For more information, please refer to the original author's blog: https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98.

## Preparation

Download the pretrained yolov3 model from https://pjreddie.com/media/files/yolov3.weights. Or you can download the same file with baidu disk:

```
链接: https://pan.baidu.com/s/1nGwNtNlhBrdJfa1LZMANUg 提取码: 54za \
```

After downloading, place the file under `config` folder.

## Usage

To detect on images, run the following command:

```bash
python object_detector.py images/blueangels.jpg
```

To detect on videos, run the following command:

```bash
python object_detector.py images/demo_video.mp4
```

## References

1. YOLOv3: https://pjreddie.com/darknet/yolo/
2. Erik Lindernoren's YOLO implementation: https://github.com/eriklindernoren/PyTorch-YOLOv3
3. YOLO paper: https://pjreddie.com/media/files/papers/YOLOv3.pdf
4. SORT paper: https://arxiv.org/pdf/1602.00763.pdf
5. Alex Bewley's SORT implementation: https://github.com/abewley/sort
6. Installing Python 3.6 and Torch 1.0: https://medium.com/@chrisfotache/getting-started-with-fastai-v1-the-easy-way-using-python-3-6-apt-and-pip-772386952d03
