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
import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

import lean.quantize as quantize
import lean.funcs as funcs
# from lean.train import qat_train

from mmcv import Config, DictAction
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataset,build_dataloader
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger
# from mmdet3d.utils import recursive_eval

#Additions
from mmcv.runner import  load_checkpoint,save_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.cnn import resnet
from mmcv.cnn.utils.fuse_conv_bn import _fuse_conv_bn
from pytorch_quantization.nn.modules.quant_conv import QuantConv2d, QuantConvTranspose2d


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj

def fuse_conv_bn(module):
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, QuantConv2d) or isinstance(child, nn.Conv2d): # or isinstance(child, QuantConvTranspose2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module


def load_model(cfg, checkpoint_path = None):
    model = build_model(cfg.model)
    if checkpoint_path != None:
        checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
    return model

def quantize_net(model):
    quantize.quantize_encoders_lidar_branch(model.lidar_modal_extractor.backbone)
    # model.lidar_modal_extractor.backbone = \
    #     funcs.layer_fusion_bn(model.lidar_modal_extractor.backbone) // bad
    quantize.quantize_encoders_camera_backbone_with_neck(model.img_backbone, model.img_neck)
    # import pdb; pdb.set_trace()
    # quantize.quantize_mask2map_decoder(model.pts_bbox_head)
    return model
    
def main():
    quantize.initialize()  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar="FILE", default="Mask2Map/projects/configs/mask2map/M2M_nusc_r50_full_fusion_2Phase_22n22ep.py", help="config file")
    parser.add_argument("--ckpt", default="model/rs50_fusion-30a8ee8a.pth", help="the checkpoint file to resume from")
    parser.add_argument("--calibrate_batch", type=int, default=300, help="calibrate batch")
    parser.add_argument('--eval', type=str, default='chamfer')
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument('--ptq_debug', type=bool, default=False, help='only for debug')
    
    args = parser.parse_args()
    args.ptq_only = True
    
    # configs.load(args.config, recursive=True)
    # cfg = Config(recursive_eval(configs), filename=args.config)
    
    cfg = Config.fromfile(args.config)
    deterministic = False
    
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set random seeds
    if args.seed is not None:
        print(
            f"Set random seed to {args.seed}, "
            f"deterministic mode: {deterministic}"
        )
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    distributed =False
    #Create Model
    model = load_model(cfg, checkpoint_path = args.ckpt)
    # torch.save(model, "model/resnet50/bevfusion-det_whole.pth")
    dataset_train  = build_dataset(cfg.data.train)
    data_loader_train =  build_dataloader(
        dataset_train,
        samples_per_gpu=1,  
        workers_per_gpu=1,  
        dist=distributed,
        seed=args.seed,
    )
        
    dataset_test = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset_test,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
    )
    
    ENABLE_PTQ = args.ptq_debug
    if ENABLE_PTQ:
        save_path = 'model/qat/rs50_fusion-30a8ee8a_ptq.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print('DataLoad Info:', data_loader_train.batch_size, data_loader_train.num_workers)
        
        model = quantize_net(model)
        
        # import pdb;pdb.set_trace()
        # model = fuse_conv_bn(model) # bad
        
        model = MMDataParallel(model, device_ids=[0])
        model.eval()

        ##Calibrate
        print("🔥 start calibrate 🔥 ")
        # quantize.set_quantizer_fast(model)
        
        
        quantize.calibrate_model(model, data_loader, 0, None, args.calibrate_batch)
        
        # quantize.disable_quantization(model.module.encoders.lidar.backbone.conv_input).apply()
        # quantize.disable_quantization(model.module.decoder.neck.deblocks[0][0]).apply()
        # quantize.print_quantizer_status(model)
        
        print(f"Done due to ptq only! Save checkpoint to {save_path} 🤗")
        # model.module.encoders.lidar.backbone = funcs.fuse_relu_only(model.module.encoders.lidar.backbone)
        torch.save(model, save_path)
        exit(1)

    if not ENABLE_PTQ:
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
        
        
        outputs = None
        from mmdet3d.apis import single_gpu_test
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
                print(dataset_test.evaluate(outputs, **eval_kwargs))
    
    # finish    
    return

if __name__ == "__main__":
    main()