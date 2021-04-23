'''
Usage:
    python object_detector.py images/blueangels.jpg
'''
from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import cv2


def detect_image(model, img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
        transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
        transforms.ToTensor(),
        ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = image_tensor.to(device)
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


if __name__ == '__main__':

    img_path = sys.argv[1]

    config_path='config/yolov3.cfg'
    weights_path='config/yolov3.weights'
    class_path='config/coco.names'
    img_size=416
    conf_thres=0.8
    nms_thres=0.4

    # Load model and weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Darknet(config_path, img_size=img_size)
    model.load_weights(weights_path)
    model.to(device)
    model.eval()
    classes = utils.load_classes(class_path)
    Tensor = torch.cuda.FloatTensor

    # load image and get detections
    prev_time = time.time()
    img = Image.open(img_path)
    detections = detect_image(model, img)
    inference_time = datetime.timedelta(seconds=time.time() - prev_time)
    print ('Inference Time: %s' % (inference_time))

    img = np.array(img)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        # browse detections and draw bounding boxes
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
            color = colors[int(cls_pred) % len(colors)]
            cls = classes[int(cls_pred)]
            cv2.rectangle(img, (x1, y1), (x1+box_w, y1+box_h), color, 2)
            cv2.rectangle(img, (x1, y1-35), (x1+len(cls)*18, y1), color, -1)
            cv2.putText(img, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    cv2.imwrite(img_path.replace(".jpg", "-det.jpg"), img)
    cv2.imshow('Result', img)
    cv2.waitKey(0)
