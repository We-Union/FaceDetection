from typing import Iterable, Tuple, Union
from torch import nn, optim
import torch
from torch.nn.modules.module import Module
from torchvision.ops import nms
import numpy as np

from face_detection import cig
from face_detection.utils import *

# this is the framework of the whole faster rcnn
# it generally integate all the module into a class
# the final network class will be derived from the class
class AbstractFasterRcnn(nn.Module):
    """
        Args:
        - extractor : top30 layer of the backbone(vgg16)
        - rpn : RPN network
        - roi_head : head of the roi
    """
    def __init__(self, extractor : nn.Module, rpn : nn.Module, roi_head : nn.Module, 
                loc_nor_mean : Tuple[float] = (0., 0., 0., 0.),
                loc_nor_std : Tuple[float] = (0.1, 0.1, 0.2, 0.2)
            ):
        super().__init__()
        self.extractor : nn.Module  = extractor
        self.rpn : Module           = rpn
        self.roi_head : Module      = roi_head
        self.loc_nor_mean : Tuple[float] = loc_nor_mean
        self.loc_nor_std : Tuple[float]  = loc_nor_std
        
        self.set_threshold(cig.nums_threshold, cig.small_score_threshold)

    def forward(self, x : torch.Tensor, scale : float = 1.0) -> Tuple[torch.Tensor, ...]:
        img_shape : torch.Size = x.shape[2:]

        # go the pipeline
        out                        = self.extractor(x)
        # rpn output : rpn_locs, rpn_scores, rois, roi_indices, anchor
        _, _, rois, roi_indices, _ = self.rpn(out, img_shape, scale)
        roi_cls_locs, roi_scores   = self.roi_head(out, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices
    
    # set the threshold of nms nad score to do the bbox filter
    def set_threshold(self, nms_threshold : float = None, score_threshold : float = None) -> None:
        if nms_threshold is None and score_threshold is None:
            self.nms_threshold = cig.nums_threshold
            self.score_threshold = cig.score_threshold
        else:
            self.nms_threshold = nms_threshold
            self.score_threshold = score_threshold

    def __suppress(self, cls_bbox : torch.Tensor, prob : torch.Tensor):
        bboxes, labels, scores = [], [], []
        # ignore class = 0, for 0 is the index of background
        for c in range(1, self.n_class):
            c_cls_bbox = cls_bbox.reshape([-1, self.n_class, 4])[:, c, :]
            c_prod = prob[:, c]

            # filter1 : filter the bbox according to the score
            mask = c_prod > self.score_threshold
            select_c_cls_box : torch.Tensor= c_cls_bbox[mask]
            select_c_prod : torch.Tensor = c_prod[mask]

            # filter2 : filter through the NMS
            nms_filter_indices : torch.Tensor = nms(select_c_cls_box, select_c_prod, self.nms_threshold)
            bboxes.append(select_c_cls_box[nms_filter_indices].cpu().tolist())
            lss : np.ndarray = (c - 1) * np.ones([len(nms_filter_indices)])
            labels.append(lss.tolist())
            scores.append(select_c_prod[nms_filter_indices].cpu().tolist())
        
        bboxes : np.ndarray = np.array(bboxes)
        labels : np.ndarray = np.array(labels)
        scores : np.ndarray = np.array(scores)
        return bboxes, labels, scores

    @without_gradient
    def predict(self, images : Union[Iterable, np.ndarray], sizes : Iterable = None, visualize : bool = False):
        """
        forward chain of the network
        Args:
            - images(Iterable) : shaped as [B, C, H, W] or list of tensor
            - sizes(Iterable) : sizes of the corresponding image in the images
            - visualize(bool) : whether visualize
        """
        self.eval()
        if visualize:
            self.set_threshold(cig.nums_threshold, cig.score_threshold)
            preprocess_imgs = [preprocess_image(img) for img in images]
            sizes = [img.shape[1:] for img in images]
        else:
            preprocess_imgs = images
        bboxes, labels, scores = [], [], []
        for img, size in zip(preprocess_imgs, sizes):
            # extend a dimension for the batch size
            img = safe_to_tensor(img).type(dtype=torch.float32)[None]
            if size[0] < 7 or size[1] < 7:
                raise ValueError("please note that input picture must be shaped as [B, C, H, W]")
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self.forward(img, scale=scale)
            # the default batch size is 1
            roi = safe_to_tensor(rois) / scale
            
            # transform the bbox from the featurem mapping space to the original picture mapping space
            mean : torch.Tensor = torch.tensor(self.loc_nor_mean).repeat(self.n_class)[None]
            std  : torch.Tensor = torch.tensor(self.loc_nor_std).repeat(self.n_class)[None]

            if cig.use_cuda:
                mean = mean.cuda()
                std = std.cuda()

            roi_cls_loc : torch.Tensor = roi_cls_loc * std + mean
            roi_cls_loc : torch.Tensor = roi_cls_loc.view(-1, self.n_class, 4)
            roi : torch.Tensor = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            # get the real bbox in the original picture
            cls_bbox = to_real_bbox(
                base_bbox=safe_to_numpy(roi).reshape([-1, 4]),
                offsets=safe_to_numpy(roi_cls_loc).reshape([-1, 4])
            )
            # type, reshape and clip
            cls_bbox = safe_to_tensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # note that picture matrix is stored as [B, C, H, W], which means size : [H, W]
            cls_bbox[..., 0::2] = cls_bbox[..., 0::2].clamp(min=0, max=size[1])         # clip x1, x2, width
            cls_bbox[..., 1::2] = cls_bbox[..., 1::2].clamp(min=0, max=size[0])         # clip y1, y2, height

            possibility = safe_to_tensor(roi_scores).softmax(dim=1)
            nms_bboxes, nms_labels, nms_scores = self.__suppress(cls_bbox, possibility)
            bboxes.append(nms_bboxes)
            labels.append(nms_labels)
            scores.append(nms_scores)
        
        self.set_threshold(0.3, 0.05)
        self.train()
        return bboxes, labels, scores

    def set_optimizer(self):
        """
            set optimizer for the class
        """
        lr = cig.learning_rate
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': cig.weight_decay}]
        if cig.optimizer_name == "Adam":
            self.optimizer = optim.Adam(params)
        else:
            self.optimizer = optim.SGD(params, momentum=0.9)
        return self.optimizer
    
    def __decay_lr(self, decay=cig.learning_rate_decay):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= decay
        return self.optimizer
        
    @property
    def n_class(self):
        return self.roi_head.n_class