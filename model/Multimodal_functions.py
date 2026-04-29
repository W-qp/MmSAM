import torch
from torch import nn
from torch.nn import functional as F
from model.MoE import MoE, Soft_MoE
import math


class Fuction(nn.Module):
    def __init__(self,
                 n_classes,
                 n_branch=None,
                 in_chans=3,
                 topk=1,
                 img_size=1024,
                 decoder_dim=64,
                 modal_dim=48,
                 modals_out_dim=128,
                 drop=0.,
                 fpn=False,
                 aux=False):
        super(Fuction, self).__init__()
        self.n_branch = len(in_chans) if isinstance(in_chans, tuple) else 1 if n_branch is None else n_branch
        if self.n_branch > 1:
            self.aux = aux
            self.modal_encoders = Modal_encoders(n_branch=self.n_branch,
                                                 in_chans=in_chans,
                                                 topk=topk,
                                                 out_chans=modals_out_dim,
                                                 patch_size=2 ** int(math.log2(img_size // 128)),
                                                 dim=modal_dim,
                                                 drop=drop)
            chans = 256 + modals_out_dim
            self.fuse = nn.Sequential(
                # CA(chans),
                nn.Conv2d(chans, decoder_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(True)
            )
            if self.aux:
                self.aux_decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.Conv2d(modals_out_dim, decoder_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(decoder_dim),
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(decoder_dim),
                    nn.ReLU(True),
                    nn.Conv2d(decoder_dim, n_classes, kernel_size=1),
                )
        else:
            self.aux = False
            chans = 256
            self.fuse = nn.Sequential(
                nn.Conv2d(chans, decoder_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(True)
            )

        self.decoder = Decoder(n_classes=n_classes, decoder_dim=decoder_dim, img_size=img_size, fpn=fpn)

    def forward(self, rgb: torch.tensor, x: list, fpnx=None):
        if self.n_branch > 1:
            x = self.modal_encoders(x)
            aux_loss = self.aux if self.training else False
            if aux_loss:
                aux = self.aux_decoder(x)
                aux = F.interpolate(aux, scale_factor=4, mode='bilinear', align_corners=True)
                x = self.fuse(torch.cat([rgb, x], dim=1))
                x = self.decoder(x, fpnx)
                return x, aux
            else:
                x = self.fuse(torch.cat([rgb, x], dim=1))
                x = self.decoder(x, fpnx)
                return x
        else:
            x = self.fuse(rgb)
            x = self.decoder(x, fpnx)
            return x


class Modal_encoders(nn.Module):
    def __init__(self,
                 n_branch,
                 in_chans,
                 topk,
                 out_chans=64,
                 patch_size=4,
                 dim=32,
                 drop=0.,
                 heads=2):
        super(Modal_encoders, self).__init__()
        in_chans = in_chans if isinstance(in_chans, tuple) else tuple([in_chans for _ in range(n_branch)])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modal_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_chans[i], dim, kernel_size=patch_size, stride=patch_size),
                nn.BatchNorm2d(dim),
                MoE(dim, dim, topk)
            ) for i in range(n_branch)
        ])

        smooth_chans = int(dim * n_branch)
        self.moe = Soft_MoE(smooth_chans, out_chans, n_branch, heads, drop)

    def forward(self, x):
        modals = []
        for i, layer in enumerate(self.modal_encoders):
            x_modal = layer(x[i])
            modals.append(x_modal)
        x = self.moe(torch.cat(modals, dim=1))
        return x


class Upsample(nn.Module):
    def __init__(self, in_chans, out_chans, mode='bilinear'):
        super(Upsample, self).__init__()
        if mode == 'conv':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_chans),
            )
        else:
            self.up = nn.Upsample(scale_factor=2, mode=mode)

    def forward(self, x):
        return self.up(x)


class CA(nn.Module):
    def __init__(self, in_chans, ratio=8, act_layer=nn.ReLU):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_chans, in_chans // ratio, kernel_size=1, bias=False),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer(),
            nn.Conv2d(in_chans // ratio, in_chans, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(x)


class Decoder(nn.Module):
    def __init__(self, n_classes, decoder_dim, img_size, mode='conv', fpn=False):
        super(Decoder, self).__init__()
        self.fpn = fpn
        self.size = img_size
        dims = [decoder_dim for _ in range(int(math.log2(img_size // 64)))]
        if fpn:
            self.bridges = nn.ModuleList([
                nn.Sequential(nn.ReLU(True), nn.Conv2d(256, dims[1], kernel_size=1)),
                nn.Sequential(nn.ReLU(True), nn.Conv2d(256, dims[2], kernel_size=1))
            ])

        self.layers = nn.ModuleList([
            nn.Sequential(
                Upsample(dims[i], dims[i + 1], mode),
                nn.Conv2d(dims[i + 1], dims[i + 1], kernel_size=3, padding=1),
                nn.GELU()
            ) for i in range(len(dims) - 1)
        ])
        self.out = nn.Conv2d(dims[-1], n_classes, kernel_size=1)

    def forward(self, x, fpnx=None):
        if self.fpn:
            xs = self.bridges[0](fpnx[1]), self.bridges[1](fpnx[0])
            for i, layer in enumerate(self.layers):
                if i < 2:
                    x = layer(x) + xs[i]
                else:
                    x = layer(x)

        else:
            for layer in self.layers:
                x = layer(x)

        x = self.out(x)
        x = F.interpolate(x, size=[self.size, self.size], mode='bilinear', align_corners=True)
        return x
