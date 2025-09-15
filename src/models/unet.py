import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import AdapterBlock


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UnetWithControl(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=64, num_levels=4, freeze_backbone=False):
        super(UnetWithControl, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        chs = [base_ch * (2 ** i) for i in range(num_levels)] # [64, 128, 256, 512]

        # encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, c in enumerate(chs):
            inc = in_ch if i == 0 else chs[i - 1]
            self.enc_blocks.append(ConvBlock(inc, c))
            if i < num_levels - 1:
                self.downs.append(nn.MaxPool2d(2))

        # bottleneck
        self.bottleneck = ConvBlock(chs[-1], chs[-1] * 2)

        # decoder
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(num_levels)):
            inc = (chs[i] * 2) if i == num_levels - 1 else chs[i+1]
            self.upconvs.append(nn.ConvTranspose2d((chs[i]*2 if i==num_levels-1 else chs[i+1]), chs[i], kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock(chs[i]*2, chs[i]))

        # final
        self.final_conv = nn.Sequential(
            nn.Conv2d(chs[0], chs[0], kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(chs[0], out_ch, kernel_size=1, stride=1, padding=0),
        )

        # adapters
        self.adapters_enc = nn.ModuleList([AdapterBlock(in_ch=1, out_ch=c) for c in chs])
        self.adapters_dec = nn.ModuleList([AdapterBlock(in_ch=1, out_ch=chs[i]*2) for i in reversed(range(num_levels))])

        if freeze_backbone:
            for p in self.enc_blocks.parameters():
                p.requires_grad = False
            for p in self.bottleneck.parameters():
                p.requires_grad = False
            for p in self.upconvs.parameters():
                p.requires_grad = False
            for p in self.dec_blocks.parameters():
                p.requires_grad = False
            for p in self.final_conv.parameters():
                p.requires_grad = False

    def forward(self, x_prior, control_map):
        # x_prior: [batch_size, 3, H, W]
        # control_map: [batch_size, 1, H, W]
        enc_feats = []
        x = x_prior

        # encoder
        for i, enc in enumerate(self.enc_blocks):
            x = enc(x)

            add = self.adapters_enc[i](control_map, x.shape)
            x = x + add

            enc_feats.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)

        # bottleneck
        x = self.bottleneck(x)

        # decoder
        for i in range(len(self.upconvs)):
            up = self.upconvs[i](x)

            skip = enc_feats[-1-i]

            if up.shape[-2:] != skip.shape[-2:]:
                up = F.interpolate(up, skip.shape[-2:], mode='bilinear', align_corners=False)

            x = torch.cat([up, skip], dim=1)

            add = self.adapters_dec[i](control_map, x.shape)

            x = x + add
            x = self.dec_blocks[i](x)

        out = self.final_conv(x)
        out = torch.tanh(out)
        return out