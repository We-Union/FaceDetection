from face_detection.Config import cig

from face_detection.utils.bbox import ellipse_to_Rectangle_label
from face_detection.utils.bbox import draw_one_boundingbox
from face_detection.utils.bbox import visualise_one_sample
from face_detection.utils.bbox import to_real_bbox
from face_detection.utils.bbox import to_offsets
from face_detection.utils.bbox import enumrate_all_shift_anchor
from face_detection.utils.bbox import generate_anchor_base

from face_detection.utils.FDDB import get_all_ellipseList_path
from face_detection.utils.FDDB import get_mate_data
from face_detection.utils.FDDB import preprocess_image

from face_detection.utils.initialize import init_with_normal
from face_detection.utils.initialize import without_gradient

from face_detection.utils.transform import safe_to_numpy
from face_detection.utils.transform import safe_to_tensor
from face_detection.utils.transform import AnchorTargetCreator
from face_detection.utils.transform import ProposalCreator
from face_detection.utils.transform import ProposalTargetCreator

from face_detection.utils.loss import smooth_l1_loss
from face_detection.utils.loss import fast_rcnn_loc_loss

__version__ = cig.Version