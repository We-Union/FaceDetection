from typing import List
import os, json
import numpy as np
from skimage.transform import resize as img_resize
from torchvision import transforms
import torch

from face_detection.utils.transform import safe_to_numpy
from face_detection.utils.bbox import ellipse_to_Rectangle_label

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

# preprocess the image to get the same size after extraction
def preprocess_image(img : np.ndarray, min_size : int = 600, max_size : int = 1000) -> np.ndarray:
    """
        input img must be shaped as [C, W, H]
    """
    img = safe_to_numpy(img)
    if img.shape[-1] in [3, 4]:
        img = np.transpose(img, axes=[2, 0, 1])
    C, H, W = img.shape
    # scale into the frame
    scale = min(min_size / min(H, W), max_size / max(H, W))
    img = img / 255
    img = img_resize(image=img, output_shape=[C, H * scale, W * scale], mode="reflect", anti_aliasing=False)
    # then normalize the picture
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    img : torch.Tensor = normalize(torch.tensor(img))
    return img.numpy()