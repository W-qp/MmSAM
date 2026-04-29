from sam2.build_sam import build_sam2
from torch import nn
from torch.nn import functional as F
from model.LoRA import LoRA_Sam2_Encoder as loraE2
from model.LoRA import Sam2_Adapter_Encoder
from model.Multimodal_functions import Fuction


class FT_SAM2(nn.Module):
    """
    Model based on SAM2's encoder.
    """

    def __init__(self,
                 n_classes,
                 img_size,
                 n_branch=None,
                 in_chans=3,
                 topk=1,
                 decoder_dim=64,
                 model_type: str = 's',  # using 't', 's', 'b+' or 'l'
                 ckpt_path: str = "./weights/",
                 modal_dim=64,
                 modals_out_dim=128,
                 drop=0.,
                 aux=False):
        super(FT_SAM2, self).__init__()
        self.n_classes = n_classes
        self.aux = aux
        self.fpn = False

        if model_type == "t":
            ckpt_path += "sam2_hiera_tiny.pt"
        elif model_type == "s":
            ckpt_path += "sam2_hiera_small.pt"
        elif model_type == "b+":
            ckpt_path += "sam2_hiera_base_plus.pt"
        elif model_type == "l":
            ckpt_path += "sam2_hiera_large.pt"
        model_type = f"configs/sam2/sam2_hiera_{model_type}.yaml"

        self.image_encoder = build_sam2(model_type, ckpt_path).image_encoder
        for params in self.image_encoder.parameters():
            params.requires_grad = False

        self.mdecoder = Fuction(n_classes=n_classes,
                                n_branch=n_branch,
                                in_chans=in_chans,
                                topk=topk,
                                img_size=img_size,
                                decoder_dim=decoder_dim,
                                modal_dim=modal_dim,
                                modals_out_dim=modals_out_dim,
                                drop=drop,
                                aux=self.aux)

    def forward(self, x):
        x = x if isinstance(x, tuple and list) else tuple([x])
        rgb = F.interpolate(x[0], size=[1024, 1024], mode='bilinear', align_corners=True)
        rgb = self.image_encoder(rgb)['backbone_fpn']
        fpnx = rgb if self.fpn else None
        x = self.mdecoder(rgb[-1], x, fpnx)
        return x


class Adapter_SAM2(nn.Module):
    """
    Model based on Adapter-SAM2's encoder.
    """

    def __init__(self,
                 n_classes,
                 img_size,
                 n_branch=None,
                 in_chans=3,
                 topk=1,
                 lora_r=8,
                 lora_step: int = 1,
                 decoder_dim=64,
                 model_type: str = 's',  # using 't', 's', 'b+' or 'l'
                 ckpt_path: str = "./weights/",
                 modal_dim=64,
                 modals_out_dim=128,
                 drop=0.,
                 fpn=False,
                 aux=False):
        super(Adapter_SAM2, self).__init__()
        self.n_classes = n_classes
        self.aux = aux
        self.fpn = fpn

        if model_type == "t":
            ckpt_path += "sam2_hiera_tiny.pt"
        elif model_type == "s":
            ckpt_path += "sam2_hiera_small.pt"
        elif model_type == "b+":
            ckpt_path += "sam2_hiera_base_plus.pt"
        elif model_type == "l":
            ckpt_path += "sam2_hiera_large.pt"
        model_type = f"configs/sam2/sam2_hiera_{model_type}.yaml"

        self.image_encoder = Sam2_Adapter_Encoder(lora_r, lora_step, model_type, ckpt_path)
        self.mdecoder = Fuction(n_classes=n_classes,
                                n_branch=n_branch,
                                in_chans=in_chans,
                                topk=topk,
                                img_size=img_size,
                                decoder_dim=decoder_dim,
                                modal_dim=modal_dim,
                                modals_out_dim=modals_out_dim,
                                drop=drop,
                                fpn=fpn,
                                aux=self.aux)

    def forward(self, x):
        x = x if isinstance(x, tuple and list) else tuple([x])
        rgb = F.interpolate(x[0], size=[1024, 1024], mode='bilinear', align_corners=True)
        rgb = self.image_encoder(rgb)['backbone_fpn']
        fpnx = rgb if self.fpn else None
        x = self.mdecoder(rgb[-1], x, fpnx)
        return x


class MLoRA_SAM2(nn.Module):
    """
    Model based on LoRA-SAM2's encoder.
    """

    def __init__(self,
                 n_classes,
                 img_size,
                 n_branch=None,
                 in_chans=3,
                 topk=1,
                 lora_r: int = 8,
                 lora_step: int = 1,
                 decoder_dim=64,
                 model_type: str = 'b+',  # using 't', 's', 'b+' or 'l'
                 ckpt_path: str = "./weights/",
                 modal_dim=64,
                 modals_out_dim=128,
                 drop=0.,
                 fpn=False,
                 aux=False):
        super(MLoRA_SAM2, self).__init__()
        self.n_classes = n_classes
        self.aux = aux
        self.fpn = fpn

        if model_type == "t":
            ckpt_path += "sam2_hiera_tiny.pt"
        elif model_type == "s":
            ckpt_path += "sam2_hiera_small.pt"
        elif model_type == "b+":
            ckpt_path += "sam2_hiera_base_plus.pt"
        elif model_type == "l":
            ckpt_path += "sam2_hiera_large.pt"
        model_type = f"configs/sam2/sam2_hiera_{model_type}.yaml"

        self.image_encoder = loraE2(lora_r, lora_step, model_type, ckpt_path)
        self.mdecoder = Fuction(n_classes=n_classes,
                                n_branch=n_branch,
                                in_chans=in_chans,
                                topk=topk,
                                img_size=img_size,
                                decoder_dim=decoder_dim,
                                modal_dim=modal_dim,
                                modals_out_dim=modals_out_dim,
                                drop=drop,
                                fpn=fpn,
                                aux=self.aux)

    def forward(self, x):
        x = x if isinstance(x, tuple and list) else tuple([x])
        rgb = F.interpolate(x[0], size=[1024, 1024], mode='bilinear', align_corners=True)
        rgb = self.image_encoder(rgb)['backbone_fpn']
        fpnx = rgb if self.fpn else None
        x = self.mdecoder(rgb[-1], x, fpnx)
        return x


class BitFit_SAM2(nn.Module):
    """
    Model based on BitFit-SAM2's encoder.
    """

    def __init__(self,
                 n_classes,
                 img_size,
                 n_branch=None,
                 in_chans=3,
                 topk=1,
                 decoder_dim=64,
                 model_type: str = 't',  # using 't', 's', 'b+' or 'l'
                 ckpt_path: str = "./weights/",
                 modal_dim=64,
                 modals_out_dim=128,
                 drop=0.,
                 fpn=False,
                 aux=False):
        super(BitFit_SAM2, self).__init__()
        self.n_classes = n_classes
        self.aux = aux
        self.fpn = fpn

        if model_type == "t":
            ckpt_path += "sam2_hiera_tiny.pt"
        elif model_type == "s":
            ckpt_path += "sam2_hiera_small.pt"
        elif model_type == "b+":
            ckpt_path += "sam2_hiera_base_plus.pt"
        elif model_type == "l":
            ckpt_path += "sam2_hiera_large.pt"
        model_type = f"configs/sam2/sam2_hiera_{model_type}.yaml"

        self.image_encoder = build_sam2(model_type, ckpt_path).image_encoder
        for params in self.image_encoder.parameters():
            params.requires_grad = False
        for name, param in self.image_encoder.named_parameters():
            if 'bias' in name:
                param.requires_grad = True

        self.mdecoder = Fuction(n_classes=n_classes,
                                n_branch=n_branch,
                                in_chans=in_chans,
                                topk=topk,
                                img_size=img_size,
                                decoder_dim=decoder_dim,
                                modal_dim=modal_dim,
                                modals_out_dim=modals_out_dim,
                                drop=drop,
                                fpn=fpn,
                                aux=self.aux)

    def forward(self, x):
        x = x if isinstance(x, tuple and list) else tuple([x])
        rgb = F.interpolate(x[0], size=[1024, 1024], mode='bilinear', align_corners=True)
        rgb = self.image_encoder(rgb)['backbone_fpn']
        fpnx = rgb if self.fpn else None
        x = self.mdecoder(rgb[-1], x, fpnx)
        return x
