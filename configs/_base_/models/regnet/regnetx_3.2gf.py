# model settings
model = dict(type='ImageClassifier',
             backbone=dict(type='RegNet', arch='regnetx_3.2gf'),
             neck=dict(type='GlobalAveragePooling'),
             head=dict(
                 type='LinearClsHead',
                 num_classes=1000,
                 in_channels=1008,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, 5),
             ))
