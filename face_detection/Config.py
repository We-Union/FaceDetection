import torch
import prettytable as pt


class Config:
    Version : str = "0.0.0"
    use_cuda = torch.has_cuda

    epoch : int = 10
    batch_size : int = 1            # NOTE: DON'T CHANGE THE VALUE!!!

    # this is the scale field in the preprocess of the image
    min_size : int = 600
    max_size : int = 1000

    freeze_layer_num : int = 5              # number of the layers that will stop to update in the feature layers in the vgg16 backbone
    use_classifier_drop : bool = False      # whether use the drop layer in the classifier layer in backbone

    negative_value : float = 0.3  # anchors with an IoU lower than this value will be assigned to negative labels
    positive_value : float = 0.7  # anchors with an IoU higher than this value will be assigned to positive labels 

    nums_threshold : float = 0.3
    score_threshold : float = 0.7

    small_score_threshold : float = 0.05

    roi_sigma : float = 1.0
    rpn_sigma : float = 3.0
    
    learning_rate : float = 1e-3
    weight_decay : float = 5e-4
    learning_rate_decay : float = 0.1
    optimizer_name : str = "SGD"

    model_path : str = None         # for persistent training
    
    # parameters for the drawing of bounding box
    BBOX_MARGIN_DICT : dict = {
        "color" : "c-",
        "width" : 5,
        "alpha" : 0.8
    }

    EPOCH_LOOP_TQDM = {
        "ncols" : 90,        # max length of progressbar
        "desc" : "Start Epoch Loop...",
    }

    BATCH_LOOP_TQDM = {
        "ncols" : 90,        # max length of progressbar
        "desc" : "Start New Batch Loop...",
    }

    def load_dict(self, **kwargs):
        _dict = self.state_dict()
        for k, v in kwargs.items():
            if k not in _dict:
                raise RuntimeError(f"Unknown Argument:{k}!!!")
            setattr(self, k, v)

        # this is a new dict
        user_dict = self.state_dict()
        tb = pt.PrettyTable()
        tb.field_names = ["Parameter", "Value"]
        print("\033[32mYour Configure:\033[0m")
        tb.add_rows(user_dict.items())
    
        print(tb)

    def state_dict(self):
        ignore_list = ["state_dict", "load_dict", "BBOX_MARGIN_DICT", "EPOCH_LOOP_TQDM", "BATCH_LOOP_TQDM"]
        return {key : getattr(self, key) for key in Config.__dict__.keys() if key not in ignore_list and not key.startswith("__")}

# namespace of the config
cig = Config()