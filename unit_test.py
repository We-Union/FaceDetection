from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import savez_compressed
import torch
import sys
import time

from torchvision.ops.roi_pool import RoIPool

from face_detection.utils import ellipse_to_Rectangle_label, draw_one_boundingbox
from face_detection.utils import generate_anchor_base, preprocess_image
from face_detection.utils.FDDB import preprocess_image
from face_detection.utils.transform import safe_to_tensor
from face_detection.Config import cig 
from face_detection import FasterRcnn

def time_count(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        temp = f(*args, **kwargs)
        print("\033[32mcost time\033[0m:", round(time.time() - start, 3), "\033[33msecond(s)\033[0m")
        return temp
    return wrapper

def test_png():
    img = Image.open("./test.png")
    img = np.array(img)
    print(img.shape)

def test_label():
    img_path = "./data/2002/08/11/big/img_591.jpg"
    x = 269.693400 
    y = 161.781200
    a = 123.583300
    b = 85.549500
    img = Image.open(img_path)
    img = np.array(img)
    plt.imshow(img)
    x1, y1, x2, y2 = ellipse_to_Rectangle_label(x, y, a, b)
    draw_one_boundingbox(x1, y1, x2, y2, color="c-", width=5, alpha=0.9)
    plt.show()

@time_count
def test_generate_anchor_base():
    anchors = generate_anchor_base()
    print(anchors)

@time_count
def test_faster_rcnn(visual : bool = False):
    img_path = "./data/2002/08/11/big/img_591.jpg"
    img = Image.open(img_path)
    img = np.array(img).transpose([2, 0, 1])
    
    # picture must be shaped as [B, C, H, W]
    model = FasterRcnn(
        n_fg_class=1
    )
    bboxes, labels, scores = model.predict(
        images=[img], 
        visualize=True
    )

    bbox = bboxes[0].ravel()
    label = labels[0].ravel()
    score = scores[0].ravel()
    print(bbox)
    print(label)
    print(score)

    if visual and bbox.size:
        plt.style.use("default")
        plt.imshow(img.transpose([1, 2, 0]))
        draw_one_boundingbox(
            x1=bbox[0],
            y1=bbox[1],
            x2=bbox[2],
            y2=bbox[3],
            **cig.BBOX_MARGIN_DICT
        )

        plt.show()    

if __name__ == "__main__":
    test_faster_rcnn(True)
