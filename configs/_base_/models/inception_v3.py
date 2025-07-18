# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='InceptionV3', num_classes=1000, aux_logits=False),
    neck=None,
    head=dict(type='ClsHead',
              loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
              topk=(1, 5)),
)
