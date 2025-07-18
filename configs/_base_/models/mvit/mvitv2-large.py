model = dict(type='ImageClassifier',
             backbone=dict(type='MViT',
                           arch='large',
                           drop_path_rate=0.5,
                           dim_mul_in_attention=False),
             neck=dict(type='GlobalAveragePooling'),
             head=dict(
                 type='LinearClsHead',
                 in_channels=1152,
                 num_classes=1000,
                 loss=dict(type='LabelSmoothLoss',
                           label_smooth_val=0.1,
                           mode='original'),
             ),
             init_cfg=[
                 dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
                 dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
             ],
             train_cfg=dict(augments=[
                 dict(type='Mixup', alpha=0.8),
                 dict(type='CutMix', alpha=1.0)
             ]))
