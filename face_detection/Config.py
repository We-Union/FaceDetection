import torch

class Config:
    Version : str = "0.0.0"
    use_cuda = torch.has_cuda

    freeze_layer_num : int = 10             # number of the layers that will stop to update in the feature layers in the vgg16 backbone
    use_classifier_drop : bool = False      # whether use the drop layer in the classifier layer in backbone
    classifier_drop_layer_indices = [2, 5]

    negative_value : float = 0.3  # anchors with an IoU lower than this value will be assigned to negative labels
    positive_value : float = 0.7  # anchors with an IoU higher than this value will be assigned to positive labels 

    nums_threshold : float = 0.3
    score_threshold : float = 0.7

    small_score_threshold : float = 0.05
    
    learning_rate : float = 1e-3
    weight_decay : float = 5e-4
    learning_rate_decay : float = 0.1
    optimizer_name : str = "Adam"
    
    # parameters for the drawing of bounding box
    BBOX_MARGIN_DICT : dict = {
        "color" : "c-",
        "width" : 5,
        "alpha" : 0.8
    }

# namespace of the config
cig = Config()