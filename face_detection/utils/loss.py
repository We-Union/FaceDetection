import torch
from face_detection.Config import cig

def smooth_l1_loss(x, t, in_weight, sigma : float):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma : float):
    in_weight = torch.zeros(gt_loc.shape)
    if cig.use_cuda:
        in_weight = in_weight.cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    indices : torch.Tensor = (gt_label > 0).view(-1, 1).expand_as(in_weight)
    in_weight[indices.cuda() if cig.use_cuda else indices.cpu()] = 1
    loc_loss = smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss