from data import data
from data.data import Dataset
import torch 
import fire
import time
import os, sys
import tqdm
from typing import Dict, Tuple
from torch import nn, optim
from torch.nn import functional
from collections import namedtuple
from visdom import Visdom

from face_detection import cig
from face_detection import FasterRcnn
from face_detection.utils import *

class EasyTrainer(nn.Module): ...

LossTuple = namedtuple("LossTuple", [
    'rpn_loc_loss',
    'rpn_cls_loss',
    'roi_loc_loss',
    'roi_cls_loss',
    'total_loss'
])

def time_count(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        temp = f(*args, **kwargs)
        print("\033[32mcost time\033[0m:", round(time.time() - start, 3), "\033[33msecond(s)\033[0m")
        return temp
    return wrapper


# A single wrapper for easy training process
class EasyTrainer(nn.Module):
    def __init__(self, faster_rcnn : FasterRcnn):
        super().__init__()

        self.faster_rcnn  : FasterRcnn   = faster_rcnn
        self.loc_nor_mean : Tuple[float] = faster_rcnn.loc_nor_mean
        self.loc_nor_std  : Tuple[float] = faster_rcnn.loc_nor_std
        self.optimizer : optim = faster_rcnn.set_optimizer()

        self.rpn_sigma : float = cig.rpn_sigma
        self.roi_sigma : float = cig.roi_sigma

        # create target creater
        self.anchorTC : AnchorTargetCreator = AnchorTargetCreator()
        self.proposalTC : ProposalTargetCreator = ProposalTargetCreator()
    
    def forward(self, images : torch.Tensor, bboxes : torch.Tensor, labels : torch.Tensor, scale : float) -> LossTuple:
        if bboxes.shape[0] != 1:
            raise RuntimeError("batch_size must be 1!!!")
        
        _, _, H, W = images.shape
        feature_mapping = self.faster_rcnn.extractor(images)
        rpn_locs, rpn_scores, rois, _, anchor = self.faster_rcnn.rpn(
            x=feature_mapping,
            img_size=[H, W],
            scale=scale
        )

        # note that batch size is 1
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois
        
        # align to get the proposal target value
        sample_roi, gt_roi_loc, gt_roi_label = self.proposalTC(
            roi=roi,
            bbox=safe_to_numpy(bbox),
            label=safe_to_numpy(label),
            loc_normalize_mean=self.loc_nor_mean,
            loc_normalize_std=self.loc_nor_std
        )

        sample_roi = safe_to_tensor(sample_roi)
        gt_roi_loc = safe_to_tensor(gt_roi_loc)
        gt_roi_label = safe_to_tensor(gt_roi_label)

        # note that we do the forwarding for one data in a batch
        # so all the choosen data in one batch is the first data, whose 
        # corresponding index is 0
        sample_roi_indices = torch.zeros(len(sample_roi))

        roi_cls_loc, roi_score = self.faster_rcnn.roi_head(
            x=feature_mapping,
            rois=sample_roi,
            roi_indices=sample_roi_indices
        )

        """calculate the RPN loss"""
        gt_rpn_loc, gt_rpn_label = self.anchorTC(
            bbox=safe_to_numpy(bbox),
            anchor=anchor,
            img_size=[H, W]
        )
        gt_rpn_label : torch.Tensor = torch.LongTensor(gt_rpn_label)
        gt_rpn_loc : torch.Tensor = safe_to_tensor(gt_rpn_loc)

        rpn_loc_loss: torch.Tensor = fast_rcnn_loc_loss(
            pred_loc=rpn_loc,
            gt_loc=gt_rpn_loc,
            gt_label=gt_rpn_label.data,
            sigma=self.rpn_sigma
        )

        # remember to ignore the bbox whose tag is -1
        rpn_cls_loss : torch.Tensor = functional.cross_entropy(
            input=rpn_score,
            target=gt_rpn_label.cuda() if cig.use_cuda else gt_rpn_label.cpu(),
            ignore_index=-1     
        )

        # cut the path of gradient to reduce the cost on GPU and remove all the label is -1
        mask : torch.Tensor = gt_rpn_label > -1
        gt_rpn_label : torch.Tensor = gt_rpn_label[mask]
        rpn_score : torch.Tensor = rpn_score[mask]

        """calculate the RoI loss"""
        n : int = roi_cls_loc.shape[0]
        roi_cls_loc : torch.Tensor = roi_cls_loc.view(n, -1, 4)

        roi_loc : torch.Tensor = roi_cls_loc[
            torch.arange(0, n),
            gt_roi_label
        ].contiguous()

        roi_loc_loss : torch.Tensor = fast_rcnn_loc_loss(
            pred_loc=roi_loc,
            gt_loc=gt_roi_loc,
            gt_label=gt_roi_label,
            sigma=self.roi_sigma
        )

        roi_cls_loss = functional.cross_entropy(
            input=roi_score,
            target=gt_roi_label
        )        


        # count all the loss
        total_loss = rpn_cls_loss + rpn_loc_loss + roi_cls_loss + roi_loc_loss
        loss_tuple = LossTuple(
            rpn_loc_loss=rpn_loc_loss,
            rpn_cls_loss=rpn_cls_loss,
            roi_loc_loss=roi_loc_loss,
            roi_cls_loss=roi_cls_loss,
            total_loss=total_loss
        )
        return loss_tuple
    
    def train_one_image(self, images : torch.Tensor, bboxes : torch.Tensor, labels : torch.Tensor, scale : float) -> LossTuple:
        """
        Args
            - images : Actually it is an image, which is shaped as [1, C, H, W]
            - bboxes : GT bbox of the items, which is shaped as [1, d, 4]
            - labels : class of each bboxes, which is shaped as [1, d]
            - scale : ratio between preprocessed image and original image   
        """
        self.optimizer.zero_grad()
        loss_tuple = self.forward(
            images=images,
            bboxes=bboxes,
            labels=labels,
            scale=scale
        )
        loss_tuple.total_loss.backward()
        self.optimizer.step()
        return loss_tuple
    
    def save(self, save_path : str = None, save_optimizer : bool = True, **kwargs):
        save_dict = {
            "model" : self.faster_rcnn.state_dict(),
            "config" : cig.state_dict(),
            "optimizer" : self.optimizer.state_dict() if save_optimizer else None,
            "info" : kwargs
        }

        if save_path is None:
            local_time = time.strftime("%m%d%H%M")
            save_path = "checkpoints/fasterrcnn_{}".format(local_time)
            for value in kwargs.values():
                save_path += "_{}".format(value)
        
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        torch.save(save_dict, save_path)

    def load(self, path : str, load_optimizer : bool = True, load_config : bool = True) -> EasyTrainer:
        state_dict = torch.load(path)
        self.faster_rcnn.load_state_dict(
            state_dict=state_dict["model"] if "model" in state_dict else state_dict
        )

        if load_optimizer and "optimizer" in state_dict and state_dict["optimizer"] is not None:
            self.optimizer.load_state_dict(state_dict["optimizer"])

        if load_config and "config" in state_dict and state_dict["config"] is not None:
            cig.load_dict(state_dict["config"])
        return self
    

@time_count
def train(**kwargs):    
    # load the configer
    cig.load_dict(**kwargs)

    # create model and training wrapper
    model = FasterRcnn()
    trainer = EasyTrainer(model)
    print("\033[32m{}\033[0m".format("complete creating model and trainer"))
    if cig.use_cuda:
        trainer = trainer.cuda()
    if cig.model_path:
        trainer.load(
            path=cig.model_path,
            load_optimizer=True,
            load_config=True
        )
    # create visdom
    vis = Visdom()

    # for decay of the learning rate
    cur_lr = cig.learning_rate
    
    # create loader of dataset
    data_set = Dataset()
    epoch_iter = tqdm.tqdm(range(cig.epoch), **cig.EPOCH_LOOP_TQDM)
    for epoch in epoch_iter:
        loader = data_set.get_train_loader()
        indices = range(data_set.training_sample_num())     # for progress bar in tqdm
        index_iter = tqdm.tqdm(indices, **cig.BATCH_LOOP_TQDM)
        epoch_iter.set_description_str("\033[32mEpoch {}\033[0m".format(epoch))

        for index, (b_img, b_bbox, b_label, scales) in zip(index_iter, loader):
            scale : float = scales[0]

            loss_tuple = trainer.train_one_image(
                images=b_img,
                bboxes=b_bbox,
                labels=b_label,
                scale=scale
            )

            post_info = "\033[33m{},{},{},{},{}\033[0m".format(
                round(loss_tuple.rpn_cls_loss.item(), 2), 
                round(loss_tuple.rpn_loc_loss.item(), 2), 
                round(loss_tuple.roi_cls_loss.item(), 2), 
                round(loss_tuple.roi_loc_loss.item(), 2), 
                round(loss_tuple.total_loss.item(), 2)
            )
            
            # set prefix and suffix info for tqdm iterator
            index_iter.set_description_str("\033[32mEpoch {} complete!\033[0m".format(epoch)
                if index == (data_set.training_sample_num() - 1) else "\033[35mtraining...\033[0m")
            index_iter.set_postfix_str(post_info)

        trainer.save()

if __name__ == "__main__":
    fire.Fire(train)