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

from torchpack.utils.config import configs
from mmcv import Config
from mmcv.runner.fp16_utils import auto_fp16
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from mmcv.runner import wrap_fp16_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import lean.quantize as quantize
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

class SubclassHeadMapSeg(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent


    def forward(self, x):
        for type, head in self.parent.heads.items():
            if type == "map":
                x = head.transform(x) # can be closed for grid sampler is not op13 supported
                x = head.classifier(x)
                return torch.sigmoid(x)
            else:
                raise ValueError(f"unsupported head: {type}")
        

class SubclassFuser(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    @auto_fp16(apply_to=("features",))
    def forward(self, features):
        if self.parent.fuser is not None:
            x = self.parent.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.parent.decoder["backbone"](x)
        x = self.parent.decoder["neck"](x)
        return x[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export transfusion to onnx file")
    parser.add_argument("--ckpt", type=str, default="pretrained/qat/bevfusion_seg_ptq.pth", help="Pretrain model")
    parser.add_argument('--fp16', action= 'store_true')
    args = parser.parse_args()
    model = torch.load(args.ckpt).module
    
    suffix = "int8"
    if args.fp16:
        suffix = "fp16"
        quantize.disable_quantization(model).apply()
    
    save_root = f"pretrained/qat/seg_onnx_{suffix}"
    os.makedirs(save_root, exist_ok=True)

    model.eval()
    fuser    = SubclassFuser(model).cuda()
    headmapseg = SubclassHeadMapSeg(model).cuda()

    TensorQuantizer.use_fb_fake_quant = True
    with torch.no_grad():
        camera_features = torch.randn(1, 80, 128, 128).cuda() # change the shape
        lidar_features  = torch.randn(1, 256, 128, 128).cuda()

        fuser_onnx_path = f"{save_root}/fuser.onnx"
        torch.onnx.export(fuser, [camera_features, lidar_features], fuser_onnx_path, opset_version=13, 
            input_names=["camera", "lidar"],
            output_names=["middle"],
        )
        print(f"🚀 The export is completed. ONNX save as {fuser_onnx_path} 🤗, Have a nice day~")

        maphead_onnx_path = f"{save_root}/head.map.onnx"
        
        head_input = torch.randn(1, 512, 128, 128).cuda() # need grid_sampler to 200 * 200
        # head_input = torch.randn(1, 512, 200, 200).cuda() # only for classifier
        
        torch.onnx.export(headmapseg, head_input, f"{save_root}/head.map.onnx", opset_version=13, 
            input_names=["middle"],
            output_names=["segmentation"],
        )
        print(f"🚀 The export is completed. ONNX save as {maphead_onnx_path} 🤗, Have a nice day~")
