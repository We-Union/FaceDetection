from typing import List
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
from face_detection.Config import cig

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
def draw_one_boundingbox(x1, y1, x2, y2, color=None, width=None, alpha=None, ax=None):
    corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if ax is None:
        ax = plt.subplot(1, 1, 1)   

    rec = plt.Rectangle(
        xy=(x1, y1),
        width=x2 - x1,
        height=y2 - y1,
        linewidth=width, 
        alpha=alpha,
        color="c",
        fill=False
    )

    ax.add_patch(rec)


# draw a picture and its labels on the pyplot
def visualise_one_sample(img_path : str, labels : List[List[float]], bbox_dict : dict =cig.BBOX_MARGIN_DICT):
    img = Image.open(img_path)
    img = np.array(img)
    plt.imshow(img)
    for label in labels:
        draw_one_boundingbox(*label, **bbox_dict)
    plt.show()

# decode : transform base bbox and offsets to the real bbox in the origin picture 
def to_real_bbox(base_bbox : np.ndarray, offsets : np.ndarray) -> np.ndarray:
    """
    Args:
        base_bbox : [x1, y1, x2, y2]
        offsets   : [dx, dy, dh, dw]
    Return:
        real_bbox : [x1, y1, x2, y2]
    """
    if base_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=offsets.dtype)

    base_bbox = base_bbox.astype(base_bbox.dtype, copy=False)

    src_height = base_bbox[:, 3] - base_bbox[:, 1]
    src_width  = base_bbox[:, 2] - base_bbox[:, 0]
    src_ctr_y  = base_bbox[:, 1] + 0.5 * src_height
    src_ctr_x  = base_bbox[:, 0] + 0.5 * src_width

    dx = offsets[:, 0::4]
    dy = offsets[:, 1::4]
    dh = offsets[:, 2::4]
    dw = offsets[:, 3::4]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    real_bbox = np.zeros(offsets.shape, dtype=offsets.dtype)
    real_bbox[:, 0::4] = ctr_x - 0.5 * w
    real_bbox[:, 1::4] = ctr_y - 0.5 * h
    real_bbox[:, 2::4] = ctr_x + 0.5 * w
    real_bbox[:, 3::4] = ctr_y + 0.5 * h

    return real_bbox

# encode : transform 
def to_offsets(base_bbox : np.ndarray, real_bbox : np.ndarray) -> np.ndarray:
    """
    Args:
        base_box : [x1, y1, x2, y2]
        real_box : [x1, y1, x2, y2]
    Returns:
        offsets  : [dx, dy, dh, dw]
    """

    height = base_bbox[:, 3] - base_bbox[:, 1]
    width  = base_bbox[:, 2] - base_bbox[:, 0]
    ctr_y  = base_bbox[:, 1] + 0.5 * height
    ctr_x  = base_bbox[:, 0] + 0.5 * width

    base_height = real_bbox[:, 3] - real_bbox[:, 1]
    base_width  = real_bbox[:, 2] - real_bbox[:, 0]
    base_ctr_y  = real_bbox[:, 1] + 0.5 * base_height
    base_ctr_x  = real_bbox[:, 0] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    offsets = np.vstack((dx, dy, dh, dw)).transpose()
    return offsets

# Calculate the Intersection of Unions between bounding boxes.
def bbox_iou(bbox_a : np.ndarray, bbox_b : np.ndarray) -> np.ndarray:

    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


# generate the base bbox(anchor)
def generate_anchor_base(base_size : int = 16, ratios : list = [0.5, 1, 2], anchor_scales : list = [8, 16, 32]) -> np.ndarray:
    """
        generate 9 base bbox, which can be divided into len(anchor_scales) groups, 
        each group is similar to the others

        base_size is the reference size of the window

        returns shaped as [R, 4]
        [x1, y1, x2, y2]
    """
    px = py = base_size / 2.
    anchor_base = np.zeros([len(ratios) * len(anchor_scales), 4], dtype=np.float32)
    for i, _ in enumerate(ratios):
        sqrt_ratio = np.sqrt(ratios[i])     # optimize the calculation process
        for j, _ in enumerate(anchor_scales):
            # make sure the size
            h = base_size * anchor_scales[j] * sqrt_ratio
            w = base_size * anchor_scales[j] / sqrt_ratio
            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = px - w / 2.
            anchor_base[index, 1] = py - h / 2.
            anchor_base[index, 2] = px + w / 2.
            anchor_base[index, 3] = py + h / 2.
    return anchor_base

# list all the base bbox according to each 
def enumrate_all_shift_anchor(base_bbox : np.ndarray, stride : int, height : int, width : int) -> np.ndarray:
    sy = np.arange(0, height * stride, stride)
    sx = np.arange(0, width * stride, stride)

    sx, sy = np.meshgrid(sx, sy)
    sx = sx.ravel()
    sy = sy.ravel()
    shift = np.stack([sx, sy, sx, sy], axis=1)
    anchor = base_bbox.reshape([1, -1, 4])
    shift = shift.reshape([1, -1, 4]).transpose([1, 0, 2])
    anchor : np.ndarray = anchor + shift
    return anchor.reshape([-1, 4]).astype(np.float32)
