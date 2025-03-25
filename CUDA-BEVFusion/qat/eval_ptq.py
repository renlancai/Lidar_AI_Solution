# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys


import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import random
import numpy as np

import onnx
import torch
from onnxsim import simplify
from torchpack.utils.config import configs
from mmcv import Config, DictAction
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

from mmdet3d.datasets import build_dataset,build_dataloader
from mmdet.datasets import replace_ImageToTensor
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from torch import nn
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
import lean.quantize as quantize

from mmdet3d.apis import single_gpu_test

def parse_args():
    parser = argparse.ArgumentParser(description="Export bevfusion model")
    
    parser.add_argument("--config", metavar="FILE", default="bevfusion/configs/nuscenes/seg/fusion-bev256d2-rs50_depth_lss.yaml", help="config file")
    parser.add_argument('--ckpt', type=str, default='pretrained/bevfusion-seg_depth.pth')
    # parser.add_argument('--ckpt', type=str, default='pretrained/qat/bevfusion_seg_ptq.pth')
    parser.add_argument('--eval', type=str, default='map')
    
    
    # parser.add_argument("--config", metavar="FILE", default="bevfusion/configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml", help="config file")
    # # parser.add_argument('--ckpt', type=str, default='model/resnet50/bevfusion-det.pth') # good
    # parser.add_argument('--ckpt', type=str, default='pretrained/qat/bevfusion_ptq.pth') # nds = 0.0, desaster
    # parser.add_argument('--eval', type=str, default='bbox')
    
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    
    parser.add_argument('--fp16', action= 'store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    
    # set random seeds
    if cfg.seed is not None:
        print(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        import random
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    
    
    
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    _ = load_checkpoint(model, args.ckpt, map_location="cpu")
    
    # model  = torch.load(args.ckpt).module
    # model  = torch.load(args.ckpt)
    
    print(type(model))
    
    suffix = "int8"
    if args.fp16:
        suffix = "fp16"
        quantize.disable_quantization(model).apply()
    
    model = MMDataParallel(model, device_ids=[0])
    print(type(model))
    model.eval()
    
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
    )

    outputs = None
    outputs = single_gpu_test(model, data_loader)
    
    rank, _ = get_dist_info()
    if rank == 0:
        # import pdb; pdb.set_trace()
        kwargs = {} if args.eval_options is None else args.eval_options
        
        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))
    

if __name__ == "__main__":
    main()
