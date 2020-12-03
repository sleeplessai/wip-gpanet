from segp2cls_v0 import _conv2d_norm_leaky_relu, se_gpm_block, _resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F


class se_gpm2cls_v0_a(nn.Module):
    def __init__(self, resnet, in_channels=[2048], reduce2=[512], gpm_n=[2], se_design='se', n_classes=0):
        super(se_gpm2cls_v0_a, self).__init__()
        self._id = "se-gpm2cls_v0a"
        self._descirption = "reset50 stage_5(512-d/2048-d), 1segpm(2x2)"
        self.backbone = resnet
        self.scale = nn.ModuleList()
        self.se_gpm = nn.ModuleList()
        self.fc2cls = nn.ModuleList()

        for i in range(len(in_channels)):
            if reduce2[i] != in_channels[i]:
                self.scale.append(_conv2d_norm_leaky_relu(in_channels[i], reduce2[i], 1))
            self.se_gpm.append(se_gpm_block(reduce2[i], gpm_n[i], se_design=se_design))
            self.fc2cls.append(nn.Linear(reduce2[i], n_classes, bias=False))

    def forward(self, x):
        stages = self.backbone(x)[-1:]
        y = []
        for i in range(len(self.se_gpm)):
            itmd = stages[i]
            itmd = self.se_gpm[i](itmd)
            itmd = F.adaptive_avg_pool2d(itmd, 1)
            itmd = torch.flatten(itmd, 1)
            itmd = self.fc2cls[i](itmd)
            y.append(itmd)
        return y[0]



class se_gpm2cls_v0_b(nn.Module):
    def __init__(self, resnet, in_channels=[2048], reduce2=[512], gpm_n=[2], se_design='se', n_classes=0):
        super(se_gpm2cls_v0_b, self).__init__()
        self._id = "se-gpm2cls_v0b"
        self._descirption = "reset50 stage_3(512-d)+stage_5(512-d), 1segpm(2x2)"
        self.backbone = resnet
        self.scale = nn.ModuleList()
        self.se_gpm = nn.ModuleList()
        self.fc2cls = nn.ModuleList()

        for i in range(len(in_channels)):
            self.scale.append(_conv2d_norm_leaky_relu(in_channels[i], reduce2[i], 1))
            self.se_gpm.append(se_gpm_block(reduce2[i] * 2, gpm_n[i], se_design=se_design))
            self.fc2cls.append(nn.Linear(reduce2[i] * 2, n_classes, bias=False))

    def forward(self, x):
        stages = self.backbone(x)[-3:]
        stage3 = stages[0]
        stage3 = F.adaptive_max_pool2d(stage3, stages[-1].size()[-2:])
        stages = stages[-1:]
        y = []
        for i in range(len(self.se_gpm)):
            itmd = self.scale[i](stages[i])
            itmd = torch.cat((itmd, stage3), dim=1)
            itmd = self.se_gpm[i](itmd)
            itmd = F.adaptive_avg_pool2d(itmd, 1)
            itmd = torch.flatten(itmd, 1)
            itmd = self.fc2cls[i](itmd)
            y.append(itmd)
        return y[0]



ABLATION_MODELS = {
    '0a': se_gpm2cls_v0_a,
    '0b': se_gpm2cls_v0_b
}

if __name__ == "__main__":
    x = torch.rand((16, 3, 448, 448)).cuda()
    net_v0a = se_gpm2cls_v0_a(resnet=_resnet50(), n_classes=200).cuda()
    y = net_v0a(x)