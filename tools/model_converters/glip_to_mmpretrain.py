# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_glip(ckpt):
    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        if 'language_backbone' in k or 'backbone' not in k or 'fpn' in k:
            continue
        new_v = v
        new_k = k.replace('body.', '')
        new_k = new_k.replace('module.', '')
        if new_k.startswith('backbone.layers'):
            new_k = new_k.replace('backbone.layers', 'backbone.stages')
        if 'mlp' in new_k:
            new_k = new_k.replace('mlp.fc1', 'ffn.layers.0.0')
            new_k = new_k.replace('mlp.fc2', 'ffn.layers.1')
        elif 'attn' in new_k:
            new_k = new_k.replace('attn', 'attn.w_msa')
        elif 'patch_embed' in k:
            new_k = new_k.replace('proj', 'projection')
        elif 'downsample' in new_k:
            if 'reduction.' in k:
                new_v = correct_unfold_reduction_order(new_v)
            elif 'norm.' in k:
                new_v = correct_unfold_norm_order(new_v)

        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained glip models to mmcls style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    weight = convert_glip(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()
