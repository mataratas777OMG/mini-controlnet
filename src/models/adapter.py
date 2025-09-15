import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class AdapterBlock(nn.Module):
    def __init__(self, in_ch=1, out_ch=64):
        super(AdapterBlock, self).__init__()
        self.net = nn.Sequential(
            conv3x3(in_ch, max(32, out_ch // 2)),
            nn.SiLU(),
            conv3x3(max(32, out_ch // 2), out_ch),
        )
        nn.init.constant_(self.net[-1].weight, 0)
        if self.net[-1].bias is not None:
            nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, control_map, target_shape):
        # control_map: [batch_size, 1, H, W]
        # target_shape: [batch_size, C, H, W]
        batch_size = control_map.shape[0]

        H, W = target_shape[-2], target_shape[-1]

        ctl = F.interpolate(control_map, size=(H, W), mode='bilinear', align_corners=False)
        out = self.net(ctl)
        return out
