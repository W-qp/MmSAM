from sam2.build_sam import build_sam2
import math
import torch
import torch.nn as nn
from torch import Tensor


class _LoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_k: nn.Module,
            linear_b_k: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.in_dim = qkv.in_features
        self.out_dim = qkv.out_features // 3
        self.Wq = nn.Parameter(torch.ones([1]), requires_grad=True)
        self.Wk = nn.Parameter(torch.ones([1]), requires_grad=True)
        self.Wv = nn.Parameter(torch.ones([1]), requires_grad=True)

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_k = self.linear_b_k(self.linear_a_k(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.out_dim] += new_q * self.Wq
        qkv[:, :, :, self.out_dim:-self.out_dim] += new_k * self.Wk
        qkv[:, :, :, -self.out_dim:] += new_v * self.Wv
        return qkv


class _Adapter(nn.Module):
    def __init__(
            self,
            blk: nn.Module,
            linear_a: nn.Module,
            act_layer: nn.GELU,
            linear_b: nn.Module
    ):
        super().__init__()
        self.blk = blk
        self.linear_a = linear_a
        self.gelu = act_layer
        self.linear_b = linear_b

    def forward(self, x):
        x = self.blk(x) + self.linear_b(self.gelu(self.linear_a(x)))
        return x


class LoRA_Sam2_Encoder(nn.Module):
    def __init__(self,
                 r: int,
                 step: int = 1,
                 model_type: str = "./sam2_configs/sam2_hiera_s.yaml",
                 ckpt_path: str = "./weights/sam2_hiera_small.pt",
                 lora_layer=None):
        super(LoRA_Sam2_Encoder, self).__init__()
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        sam2_imencoder = build_sam2(model_type, ckpt_path).image_encoder
        for params in sam2_imencoder.parameters():
            params.requires_grad = False
        for p in sam2_imencoder.trunk.patch_embed.parameters():
            p.requires_grad = True

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam2_imencoder.trunk.blocks)))
        self.w_As = []
        self.w_Bs = []

        for t_layer_i, blk in enumerate(sam2_imencoder.trunk.blocks):
            # If we only want few lora layer instead of all
            # if t_layer_i not in self.lora_layer:
            if t_layer_i % step != 0:
                continue
            w_qkv_linear = blk.attn.qkv
            self.in_dim = w_qkv_linear.in_features
            self.out_dim = w_qkv_linear.out_features // 3
            w_a_linear_q = nn.Linear(self.in_dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.out_dim, bias=False)
            w_a_linear_k = nn.Linear(self.in_dim, r, bias=False)
            w_b_linear_k = nn.Linear(r, self.out_dim, bias=False)
            w_a_linear_v = nn.Linear(self.in_dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.out_dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_k)
            self.w_Bs.append(w_b_linear_k)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_k,
                w_b_linear_k,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.sam = sam2_imencoder

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.sam(x)


class Sam2_Adapter_Encoder(nn.Module):
    def __init__(self,
                 r: int = None,
                 step: int = 1,
                 model_type: str = "./sam2_configs/sam2_hiera_s.yaml",
                 ckpt_path: str = "./weights/sam2_hiera_small.pt",
                 lora_layer=None):
        super(Sam2_Adapter_Encoder, self).__init__()
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        sam2_imencoder = build_sam2(model_type, ckpt_path).image_encoder
        for params in sam2_imencoder.parameters():
            params.requires_grad = False
        for p in sam2_imencoder.trunk.patch_embed.parameters():
            p.requires_grad = True

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam2_imencoder.trunk.blocks)))
        self.w_As = []
        self.w_Bs = []

        ratio = r
        for t_layer_i, blk in enumerate(sam2_imencoder.trunk.blocks):
            # If we only want few lora layer instead of all
            # if t_layer_i not in self.lora_layer:
            if t_layer_i % step != 0:
                continue
            adapter_linear = blk.mlp
            self.dim = blk.mlp.layers[1].out_features
            r = self.dim // 8 if ratio is None else ratio  # set ratio
            w_a_linear = nn.Linear(self.dim, r, bias=False)
            w_b_linear = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear)
            self.w_Bs.append(w_b_linear)
            blk.mlp = _Adapter(
                adapter_linear,
                w_a_linear,
                nn.GELU(),
                w_b_linear
            )
        self.reset_parameters()
        self.sam = sam2_imencoder

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.sam(x)
