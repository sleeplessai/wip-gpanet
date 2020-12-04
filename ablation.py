from segp2cls_v0 import _conv2d_norm_leaky_relu, se_gpm_block, _resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        self._descirption = "reset50 stage_3(512-d) fuse to stage_5(512-d), 1segpm(2x2)"
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


class se_gpm2cls_v0_c(nn.Module):
    def __init__(self, resnet, in_channels=[1024, 2048], reduce2=[512, 512], gpm_n=[2], se_design='se', n_classes=0):
        super(se_gpm2cls_v0_c, self).__init__()
        self._id = "se-gpm2cls_v0c"
        self._descirption = "reset50 stage_3(512-d)+stage_4(512_d) fuse to stage_5(512-d), 1segpm(2x2)"
        self.backbone = resnet
        self.scale = nn.ModuleList()
        self.se_gpm = nn.ModuleList()
        self.fc2cls = nn.ModuleList()

        for i in range(len(in_channels)):
            self.scale.append(_conv2d_norm_leaky_relu(in_channels[i], reduce2[i], 1))
        self.se_gpm.append(se_gpm_block(reduce2[0] * 3, gpm_n[0], se_design=se_design)) # 512x3=1536-d
        self.fc2cls.append(nn.Linear(reduce2[0] * 3, n_classes, bias=False)) # 512x3=1536-d

    def forward(self, x):
        stages = self.backbone(x)[-3:]
        stage3 = stages[0]
        stage3 = F.adaptive_max_pool2d(stage3, stages[-1].size()[-2:])
        stage4 = self.scale[0](stages[1])   # stage-4 scaling
        stage4 = F.adaptive_max_pool2d(stage4, stages[-1].size()[-2:])
        y = self.scale[1](stages[-1])  # stage-5 scaling
        y = torch.cat((y, stage4, stage3), dim=1)
        y = self.se_gpm[0](y)
        y = F.adaptive_avg_pool2d(y, 1)   # todo: maxpool or avgpool?
        y = torch.flatten(y, 1)
        y = self.fc2cls[0](y)
        return y


class se_gpm2cls_v0_d(nn.Module):
    def __init__(self, resnet, in_channels=[256, 512, 1024, 2048], reduce2=[512, 512], gpm_n=[2], se_design='se', n_classes=0):
        super(se_gpm2cls_v0_d, self).__init__()
        self._id = "se-gpm2cls_v0d"
        self._descirption = "reset50 maxpool{s2,s3,512ds4} fuse to 512ds5, 1segpm(2x2)"
        self.backbone = resnet
        self.scale = nn.ModuleList()
        self.se_gpm = nn.ModuleList()
        self.fc2cls = nn.ModuleList()

        self.scale.append(_conv2d_norm_leaky_relu(in_channels[-2], reduce2[0], 1))
        self.scale.append(_conv2d_norm_leaky_relu(in_channels[-1], reduce2[1], 1))
        # print(self.scale)
        # for i in range(len(in_channels)):
        self.se_gpm.append(se_gpm_block(np.sum(in_channels[:-1]), gpm_n[0], se_design=se_design)) # 512x3=1536-d
        self.fc2cls.append(nn.Linear(np.sum(in_channels[:-1]), n_classes, bias=False)) # 512x3=1536-d
        # print(self.fc2cls)

    def forward(self, x):
        stages = self.backbone(x)[1:]
        stages[-2] = self.scale[0](stages[-2])  # stage-4 scaling
        stages[-1] = self.scale[1](stages[-1])  # stage-5 scaling
        for i in range(3):
            stages[i] = F.adaptive_max_pool2d(stages[i], stages[-1].size()[-2:])
            # print(stages[i].size())
        y = torch.cat(stages, dim=1)
        y = self.se_gpm[0](y)
        y = F.adaptive_avg_pool2d(y, 1)   # todo: maxpool or avgpool?
        y = torch.flatten(y, 1)
        y = self.fc2cls[0](y)
        # print(y.size())
        return y



ABLATION_MODELS = {
    '0a': se_gpm2cls_v0_a,
    '0b': se_gpm2cls_v0_b,
    '0c': se_gpm2cls_v0_c,
    '0d': se_gpm2cls_v0_d
}

if __name__ == "__main__":
    x = torch.rand((16, 3, 448, 448)).cuda()
    net_v0a = se_gpm2cls_v0_a(resnet=_resnet50(), n_classes=200).cuda()
    y = net_v0a(x)