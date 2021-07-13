from face_detection.constants import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, json
from typing import List

def ellipse_to_Rectangle_label(x : float, y : float, a : float, b : float, shamt=1.0):
    if a < b:               # make sure a is the larger one
        a, b = b, a
    # the shape of origin label file gives is ellipse, so I need to transform it into rectangle region
    x1 = x - b / shamt      
    y1 = y - a / shamt
    x2 = x + b / shamt
    y2 = y + a / shamt
    return x1, y1, x2, y2

# use plt to draw bounding box on the picture
def draw_one_boundingbox(x1, y1, x2, y2, color=None, width=None, alpha=None):
    corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    for i, _ in enumerate(corners):             # draw the line clock wise
        p1 = corners[i % len(corners)]
        p2 = corners[(i + 1) % len(corners)]
        xx = [p1[0], p2[0]]
        yy = [p1[1], p2[1]]
        plt.plot(xx, yy, color, linewidth=width, alpha=alpha)
    
# get all the origin label files' path
def get_all_ellipseList_path(file_dir : str) -> List[str]:
    result = [file_dir + "/" + file for file in os.listdir(file_dir) if "ellipse" in file]
    return result

# transform all the label file to the meta data with format of json
def get_mate_data(label_file_paths : List[str], dump_path : str):
    target_dict = {}        # final dumped data structure
    for file_path in label_file_paths:
        with open(file_path, "r", encoding="utf-8") as fp:
            line = fp.readline().strip()
            while line:
                label_num = int(fp.readline().strip())
                labels = []         # all the labels corresponding to current image
                for _ in range(label_num):
                    label_info = fp.readline().strip().split()
                    # 0, 1, 3, 4 : a, b, x, y
                    # transform to rectangle region
                    x1, y1, x2, y2 = ellipse_to_Rectangle_label(
                        x=float(label_info[3]),
                        y=float(label_info[4]),
                        a=float(label_info[0]),
                        b=float(label_info[1])
                    )
                    labels.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
                img_path = "./data/" + line + ".jpg"
                target_dict[img_path] = labels
                line = fp.readline().strip()
    with open(dump_path, "w", encoding="utf-8") as fp:
        json.dump(obj=target_dict, fp=fp, ensure_ascii=False, indent=4)


def visualise_one_sample(img_path : str, labels : List[List[float]], bounding_box_dict : dict =BOUNDONGING_BOX_MARGIN_DICT):
    img = Image.open(img_path)
    img = np.array(img)
    plt.imshow(img)
    for label in labels:
        draw_one_boundingbox(*label, **bounding_box_dict)
    plt.show()