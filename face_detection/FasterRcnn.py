from face_detection.Rpn import RPN
import torch 
from torch import nn 
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from face_detection.Config import cig
from face_detection.utils.initialize import init_with_normal
from face_detection.utils.transform import safe_to_tensor
from face_detection.AbstractFasterRcnn import AbstractFasterRcnn

# decomponent vgg16 into extractor and classifier
def get_component_from_vgg16(pretrained : bool):
    model = vgg16(pretrained=pretrained)        # download may take some time if this line is execuated initially
    """
    model.features:
    Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace=True)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace=True)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace=True)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace=True)
        (20): ReLU(inplace=True)
        (22): ReLU(inplace=True)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace=True)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace=True)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace=True)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )

    model.classifier:
    Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=1000, bias=True)
    )
    """
    
    extractor = list(model.features)[:-1]           # drop the last pooling
    classifier = list(model.classifier)[:-1]        # we will redefine the softmax

    # the pre 4 layers don't need to be updated
    for layer in extractor[:cig.freeze_layer_num]:
        for param in layer.parameters():
            param.requires_grad = False
    
    # drop all the drop layer if cig.use_classifier_drop is False
    if not cig.use_classifier_drop:
        classifier = [layer for layer in classifier if not isinstance(layer, nn.Dropout)]
    
    # transform to sequence
    extractor = nn.Sequential(*extractor)
    classifier = nn.Sequential(*classifier)
    # component = {
    #     "extractor" : extractor,
    #     "classifier" : classifier
    # }
    return extractor, classifier

class RoIHead(nn.Module):
    def __init__(self, n_class : int, roi_size : int, spatial_scale, classifier):
        """
        Args:
            - n_class : class number of foreground
            - roi_size : size of the roi after the RoI pooling, which si assigned as 7 in the original paper
            - spatial_scale
        """
        super().__init__()
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool(
            output_size=[self.roi_size, self.roi_size], 
            spatial_scale=self.spatial_scale
        )

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        init_with_normal(self.cls_loc, 0, 0.001)
        init_with_normal(self.score, 0, 0.01)

    def forward(self, x : torch.Tensor, rois : torch.Tensor, roi_indices : torch.Tensor):
        roi_indices = safe_to_tensor(roi_indices).float()
        rois = safe_to_tensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)

        # xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        # indices_and_rois =  xy_indices_and_rois.contiguous()
        # RoI pooling
        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores

class FasterRcnn(AbstractFasterRcnn):
    """
        this the final module of the whole model, which is derived from AbstractFasterRcnn
    Args:
        - n_fg_class(int) : number of class of foreground items
        - ratios(list) : ratio of width and height of the generated anchor base
        - anchor_scales(list) : scale from the base bbox
    """
    def __init__(self, n_fg_class : int = 1, ratios : list = [0.5, 1, 2], anchor_scales : list = [8, 16, 32]):
        self.stride = 16
        extractor, classifier = get_component_from_vgg16(pretrained=True)

        rpn = RPN(
            input_channel=512,
            output_channel=512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            stride=self.stride
        )

        roi_head = RoIHead(
            n_class=n_fg_class + 1,             # 1 is background
            roi_size=7,
            spatial_scale=1. / self.stride,
            classifier=classifier
        )

        super().__init__(
            extractor=extractor,
            rpn=rpn,
            roi_head=roi_head
        )
        