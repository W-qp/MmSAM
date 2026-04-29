import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MoE(nn.Module):
    def __init__(self, dim, out_chans, top_k, num_experts=4):
        super(MoE, self).__init__()
        self.out_chans = out_chans
        self.top_k = top_k
        self.router = GateRouter(dim, top_k, num_experts)
        self.experts = nn.ModuleList([Expert1(dim, out_chans),
                                      Expert2(dim, out_chans),
                                      Expert3(dim, out_chans),
                                      Expert4(dim, out_chans),])
        self.down = Downsample(out_chans, out_chans)

    def forward(self, x):
        gating_scores, indices = self.router(x)
        final_output = torch.zeros([x.shape[0], self.out_chans, 128, 128], device=x.device)
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=1)
            if expert_mask.any():
                expert_input = x[expert_mask]
                expert_output = expert(expert_input)
                weights = gating_scores[expert_mask, i].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                weighted_output = expert_output * weights
                final_output[expert_mask] += weighted_output
        return self.down(final_output)


class GateRouter(nn.Module):
    def __init__(self, dim, top_k, num_experts):
        super(GateRouter, self).__init__()
        self.top_k = top_k
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.linear = nn.Linear(dim, num_experts)
        self.r = nn.Parameter(torch.tensor(0.3), requires_grad=False)###

    def forward(self, x):
        avg_x, max_x = self.avg(x), self.max(x)
        x = avg_x * (1 - self.r) + max_x * self.r
        x = x.squeeze(-1).squeeze(-1)
        experts_scores = self.linear(x)
        top_k_logits, indices = experts_scores.topk(self.top_k, dim=-1)
        zeros = torch.full_like(experts_scores, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        experts_scores = F.softmax(sparse_logits, dim=1)
        return experts_scores, indices


class Expert1(nn.Module):
    def __init__(self, dim, out_chans):
        super(Expert1, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, out_chans, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Expert2(nn.Module):
    def __init__(self, dim, out_chans):
        super(Expert2, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, out_chans, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Expert3(nn.Module):
    def __init__(self, dim, out_chans):
        super(Expert3, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim*2, kernel_size=1),
            nn.BatchNorm2d(dim*2),
            nn.Conv2d(dim*2, out_chans, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Expert4(nn.Module):
    def __init__(self, dim, out_chans):
        super(Expert4, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, out_chans, kernel_size=3, padding=1),
            nn.ReLU(True),
            Downsample(dim, dim, mode='conv'),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.net(x)
        return x


class Basic_layer(nn.Module):
    def __init__(self, in_channels, ratio=1., n_groups=4, depth=2, dilation=1):
        super(Basic_layer, self).__init__()
        dim = int(in_channels * ratio)
        self.n = depth
        for i in range(1, depth + 1):
            conv = nn.Sequential(
                nn.Conv2d(in_channels, dim, kernel_size=3, padding=dilation, groups=n_groups, dilation=dilation),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, in_channels, kernel_size=3, padding=1),
                nn.GELU(),
            )
            setattr(self, 'conv%d' % i, conv)

    def forward(self, x):
        short_cut = x
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x) + short_cut
            short_cut = x
        return x


class Downsample(nn.Module):
    def __init__(self, in_chans, out_chans, mode='conv'):
        super(Downsample, self).__init__()
        self.mode = mode
        if mode == 'merging':
            self.down = nn.Sequential(
                nn.LayerNorm(in_chans * 4),
                nn.Linear(in_chans * 4, out_chans)
            )
        elif mode == 'maxpool':
            self.down = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
                nn.BatchNorm2d(out_chans)
            )
        else:
            self.down = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_chans),
                nn.ReLU(),
            )

    def forward(self, x):
        if self.mode == 'merging':
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)
            pad_input = (H % 2 == 1) or (W % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
            x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
            x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
            x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
            x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
            x = self.down(x).transpose(1, 2).contiguous().view(-1, C, H, W)

        else:
            x = self.down(x)
        return x


class Soft_MoE(nn.Module):
    def __init__(self,
                 in_chans,
                 out_chans,
                 n_branch,
                 n_heads=4,
                 fuse_drop=0.,
                 qkv_bias=True,
                 patch_size=4):
        super().__init__()
        self.n_heads = n_heads
        self.short_cut_conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=1),
            nn.ReLU(True),
        )

        self.q = nn.Sequential(
            nn.MaxPool2d(patch_size, patch_size),
            nn.Conv2d(in_chans, in_chans, kernel_size=1, bias=qkv_bias, groups=n_branch)
        )
        self.k = nn.Sequential(
            nn.MaxPool2d(patch_size, patch_size),
            nn.Conv2d(in_chans, in_chans, kernel_size=1, bias=qkv_bias, groups=n_branch)
        )
        self.v = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1, groups=in_chans),
            nn.ReLU(True),
            nn.Conv2d(in_chans, in_chans, kernel_size=1, bias=qkv_bias, groups=n_branch)
        )

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((n_heads, 1, 1))), requires_grad=True)

        self.cpb_mlp = nn.Sequential(nn.Linear(1, 16 * n_branch, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(16 * n_branch, n_heads, bias=False))

        coords = torch.zeros([in_chans, in_chans], dtype=torch.int64)
        for idx in range(in_chans):
            coords[idx] = torch.arange(in_chans) - idx
        relative_position_bias = coords / coords.max()
        relative_position_bias *= 8  # normalize to -8, 8
        relative_position_bias = torch.sign(relative_position_bias) * torch.log2(
            torch.abs(relative_position_bias) + 1.0) / np.log2(8)
        self.register_buffer("relative_position_bias", relative_position_bias.unsqueeze(-1))

        self.dropout = nn.Dropout(fuse_drop)
        self.norm = nn.BatchNorm2d(out_chans)
        self.softmax = nn.Softmax(dim=-1)
        self.softmax1 = nn.Softmax(dim=-1)
        self.proj = nn.Sequential(nn.Conv2d(in_chans, out_chans, kernel_size=1),
                                  nn.GELU())

    def forward(self, x):
        short_cut = x
        b, c, H, W = x.shape
        q, k, v = self.q(x).flatten(2), self.k(x).flatten(2), self.v(x).flatten(2)  # b, c, h*w
        q = q.reshape(b, c, self.n_heads, -1).permute(0, 2, 1, 3)  # b, n, c, h*w//n
        k = k.reshape(b, c, self.n_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(b, c, self.n_heads, -1).permute(0, 2, 1, 3)

        # cosine attention
        sim = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to(sim.device))).exp()
        sim = sim * logit_scale

        relative_position_bias = self.cpb_mlp(self.relative_position_bias).view(-1, self.n_heads).view(c, c, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = torch.sigmoid(relative_position_bias)
        sim = sim + relative_position_bias.unsqueeze(0)

        sim = self.softmax1(1 - self.softmax(sim))
        sim = self.dropout(sim)
        x = (sim @ v).transpose(1, 2).reshape(b, c, -1)  # b, c, h*w
        x = x.view(b, -1, H, W)  # b, c, h, w
        x = self.proj(x)
        x = self.dropout(x)
        x = self.norm(x) + self.short_cut_conv(short_cut)
        return x


if __name__ == '__main__':
    model = MoE(16, 16, 2).cuda()
    img = torch.randn([2, 16, 128, 128])
    img1 = torch.randn([1, 16, 128, 128]) - 1
    img = torch.cat([img, img1], dim=0).cuda()
    pred = model(img)
    print(pred.shape)
