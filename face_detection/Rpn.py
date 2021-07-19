from typing import Union
from face_detection.utils.transform import ProposalCreator
from face_detection.utils.bbox import generate_anchor_base
from torch import nn 
import torch 
import numpy as np

from face_detection.utils import init_with_normal
from face_detection.utils import enumrate_all_shift_anchor



class RPN(nn.Module):
    def __init__(self, input_channel : int, output_channel : int, ratios=[0.5, 1, 2],
            anchor_scales : list = [8, 16, 32], stride : int = 16, proposal_creator_params : dict = {}):
        super().__init__()
        self.anchor_base = generate_anchor_base(
            base_size=16,
            ratios=ratios,
            anchor_scales=anchor_scales
        )
        self.stride = stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, 1, 1)
        self.score = nn.Conv2d(output_channel, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(output_channel, n_anchor * 4, 1, 1, 0)
        init_with_normal(self.conv1, 0, 0.01)
        init_with_normal(self.score, 0, 0.01)
        init_with_normal(self.loc, 0, 0.01)
    
    def forward(self, x : torch.Tensor, img_size : Union[list, tuple], scale : float = 1.):
        n, _, hh, ww = x.shape
        anchor = enumrate_all_shift_anchor(
            np.array(self.anchor_base),
            self.stride, hh, ww
        )

        n_anchor = anchor.shape[0] // (hh * ww)
        h = self.conv1(x).relu()

        rpn_locs = self.loc(h)
        # UNNOTE: check whether need contiguous
        # A: Yes
        # move c channel to the last channel 
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = rpn_scores.view(n, hh, ww, n_anchor, 2).softmax(dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale
            )

            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor
    