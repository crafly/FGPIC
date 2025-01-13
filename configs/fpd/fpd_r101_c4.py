_base_ = [
    './meta-rcnn_r50_c4.py',
]
pretrained = 'open-mmlab://detectron2/resnet101_caffe'
# model settings
model = dict(
    type='FPD',
    post_rpn=True,
    pretrained=pretrained,
    backbone=dict(depth=101),
    roi_head=dict(
        type='FPDRoIHead',
        shared_head=dict(pretrained=pretrained),
        bbox_head=dict(num_classes=20, num_meta_classes=20),
        novel_class=(15, 16, 17, 18, 19),
        prototypes_distillation=dict(
            type='PrototypesDistillation',
            num_queries=5, dim=1024, num_base_cls=15),
        prototypes_assignment=dict(
            type='PrototypesAssignment',
            dim=1024, num_bg=5),
        
        prototype_aggregation=dict(
            type='PrototypeDynamicAggregation',
            in_channels=2048,
            alpha=1.0
        ),
        #加
    ),
    #
    information_coupling=dict(
        type='ConditionalInformationCouplingModule',
        in_channels=1024,
        inter_channels=1024,
        dimension=2,
        sub_sample=False,
        bn_layer=True)
    #加

)
