import copy
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

from typing import Optional, Union, Callable, List
from model.attention import (
    AttentionLayer,
    ProbAttention,
)
import uuid
from utils import logger
from model.model import MyModel
from model.local_module import LocalLayer
from model.global_module import GlobalLayer
from fvcore.nn import FlopCountAnalysis

from thop import profile


def count_flop(path_to_load):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(path_to_load, weights_only=False).to(device)

    model.eval()
    lh = torch.randn(1, 204, 21, 2).to(device=device, dtype=torch.float32)
    rh = torch.randn(1, 204, 21, 2).to(device=device, dtype=torch.float32)
    bd = torch.randn(1, 204, 12, 2).to(device=device, dtype=torch.float32)

    macs, params = profile(model, inputs=(lh, rh, bd))

    # flops = FlopCountAnalysis(model, (lh, rh, bd))
    # print("FLOPs: ", flops.total(), "GFLOPs")
    # with torch.no_grad():
    #     out = model(lh, rh, bd)
    print("FLOPs:", macs * 2 / 1e9)
    print("params:", params)
    # return flops.total() / 1e9


if __name__ == "__main__":
    module = MyModel(
        num_enc_layers=1,
        num_dec_layers=2,
        pat_dec=0,
    )

    print(f'GFLOPs: {count_flop(f"out-checkpoints/WLASL100/checkpoint_v_10.pth")}')
    print(f"Params: {sum(p.numel() for p in module.parameters() if p.requires_grad)}")
