import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def get_model(num_classes, cfg):
    # Backbone
    # Determine backbone name from config (default to resnet50)
    # Support 'resnet50', 'resnet101', 'resnext50_32x4d', 'resnext101_32x4d', 'wide_resnet101_2'
    config_backbone = cfg['model'].get('backbone', 'resnet50')
    
    if 'resnext101' in config_backbone:
        backbone_name = 'resnext101_32x8d' # Valid in current torchvision
    elif 'resnext50' in config_backbone:
        backbone_name = 'resnext50_32x4d'
    elif 'wide_resnet101' in config_backbone:
        backbone_name = 'wide_resnet101_2'
    elif 'resnet101' in config_backbone:
        backbone_name = 'resnet101'
    else:
        backbone_name = 'resnet50' # Default

    # trainable_layers: number of block layers to return trainable (0 to 5)
    backbone = resnet_fpn_backbone(backbone_name, 
                                   trainable_layers=cfg['model']['trainable_backbone_layers'], 
                                   weights='DEFAULT' if cfg['model']['pretrained'] else None)
    
    # Anchor Generator
    # Parse anchor_sizes from config
    # Config format: [[16, 32, 64, 128, 256]] (single list of scales)
    # We need to map these to 5 FPN levels: ((16,), (32,), (64,), (128,), (256,))
    raw_sizes = cfg['model']['anchor_sizes']
    
    if len(raw_sizes) == 1 and len(raw_sizes[0]) == 5:
        # If user provided a single list of 5 scales, distribute them across the 5 FPN levels
        sizes = tuple((s,) for s in raw_sizes[0])
    else:
        # Otherwise use as provided (allows manual per-level config)
        sizes = tuple(tuple(x) for x in raw_sizes)
    
    aspect_ratios = tuple(tuple(x) for x in cfg['model']['aspect_ratios']) 
    ratios = aspect_ratios * len(sizes) # duplicate ratios for each level
    
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
    
    model = FasterRCNN(backbone,
                       num_classes=num_classes, # bg + classes
                       rpn_anchor_generator=anchor_generator,
                       # RPN Parameters for high recall
                       rpn_pre_nms_top_n_train=cfg['model']['rpn_pre_nms_top_n_train'],
                       rpn_post_nms_top_n_train=cfg['model']['rpn_post_nms_top_n_train'],
                       rpn_pre_nms_top_n_test=cfg['model']['rpn_pre_nms_top_n_test'],
                       rpn_post_nms_top_n_test=cfg['model']['rpn_post_nms_top_n_test'],
                       rpn_nms_thresh=cfg['model']['rpn_nms_thresh'],
                       rpn_fg_iou_thresh=cfg['model']['rpn_fg_iou_thresh'],
                       rpn_bg_iou_thresh=cfg['model']['rpn_bg_iou_thresh'],
                       # ROI Parameters
                       box_score_thresh=cfg['model']['box_score_thresh'],
                       box_nms_thresh=cfg['model']['box_nms_thresh'],
                       box_detections_per_img=cfg['model']['box_detections_per_img']
                       )
    return model
