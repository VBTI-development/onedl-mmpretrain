# model settings
model = dict(type='ImageClassifier',
             backbone=dict(type='EfficientNet', arch='b5'),
             neck=dict(type='GlobalAveragePooling'),
             head=dict(
                 type='LinearClsHead',
                 num_classes=1000,
                 in_channels=2048,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, 5),
             ))
