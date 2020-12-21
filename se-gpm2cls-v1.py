import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from fireblast.models.resnet import resnet50, resnet18
except:
    from torchvision.models import resnet50, resnet18
import numpy as np
from aolm import attention_object_location_module


def _transparent():
    return nn.Identity()


class _resnet50(nn.Module):
    def __init__(self, pretrained=True, pth_path=None):
        super(_resnet50, self).__init__()
        if not pth_path:
            self.net = resnet50(pretrained=True)
        else:
            print(f'Load weights from {pth_path}')
            self.net = resnet50(pretrained=False)
            self.net.fc = nn.Linear(2048, 200)
            self.net.load_state_dict(torch.load(pth_path))
        self.dropout = nn.Dropout(p=0.5)
        self.stage_channels = [64, 256, 512, 1024, 2048]

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)

        x1 = self.net.maxpool(x)
        x2 = self.net.layer1(x1)
        x3 = self.net.layer2(x2)
        x4 = self.net.layer3(x3)
        x5 = self.net.layer4(x4)

        return [x1, x2, x3, x4, x5]

    def _aolm_forward(self, x, scda_stage=4):
        scda = [None, None, None, None, None]

        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)     # s1

        x = self.net.layer1(x)      # s2

        conv3_b = self.net.layer2[:-1](x)
        x = self.net.layer2[-1](conv3_b)
        conv3_c = x
        scda[2] = (conv3_c, conv3_b)    # s3

        conv4_b = self.net.layer3[:-1](x)
        x = self.net.layer3[-1](conv4_b)
        conv4_c = x
        scda[3] = (conv4_c, conv4_b)    # s4

        conv5_b = self.net.layer4[:2](x)
        x = self.net.layer4[2](conv5_b)
        conv5_c = x
        scda[4] = (conv5_c, conv5_b)    # s5

        x = self.net.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        embedding = x

        conv_c, conv_b = scda[scda_stage]
        # conv5_c from last conv layer, \
        # conv5_b is the one in front of conv5_c
        return conv_c, conv_b, embedding


class _conv2d_norm_relu(nn.Module):
    _negative_slope = .5 / np.e

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(_conv2d_norm_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(self._negative_slope, inplace=True)

    def forward(self, x, activate=True):
        x = self.conv(x)
        x = self.norm(x)
        if activate: x = self.relu(x)
        return x


class global_perception(nn.Module):
    def __init__(self, n, in_channels, stride=1):
        super(global_perception, self).__init__()
        self.n = n
        self.convs = nn.ModuleList()
        for _ in range(n * n):
            # set groups conv to reduce parameters
            self.convs.append(_conv2d_norm_relu(
                n * n * in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels))

    def forward(self, x):
        # destruction: reshaping
        xr = []
        p = torch.chunk(x, self.n, -2)
        for o in p:
            q = list(torch.chunk(o, self.n, -1))
            xr += q
        xr = torch.cat(xr, 1)
        # construction: restoring
        q, fr = [], []
        for i, conv in enumerate(self.convs):
            f = conv(xr)
            # f = xr[:, i:(i + 1), :, :]
            q.append(f)
            if len(q) == self.n:
                p = torch.cat(q, -1)
                q = []
                fr.append(p)
        x = torch.cat(fr, -2)
        return x


class se_block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(se_block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class se_gpm_block(nn.Module):
    def __init__(self, in_channels, n, se_design='se', se_reduction=16):
        super(se_gpm_block, self).__init__()
        assert se_design in ['se', 'pre', 'post', 'identity']
        self.se_design = se_design
        self.se = se_block(in_channels, se_reduction)
        self.gpm = global_perception(n, in_channels)
        self.relu = nn.LeakyReLU(_conv2d_norm_relu._negative_slope, inplace=True)

    def forward(self, x):
        identity = x
        if self.se_design == 'se':
            x = self.gpm(x)
            x = self.se(x)
            x += identity
        elif self.se_design == 'pre':
            x = self.se(x)
            x = self.gpm(x)
            x += identity
        elif self.se_design == 'post':
            x = self.gpm(x)
            x += identity
            x = self.se(x)
        elif self.se_design == 'identity':
            x = self.se(identity) + self.gpm(identity)
        x = self.relu(x)
        return x


class scaling_layer(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], out_channels=[512, 512, 512], transparent=False):
        super(scaling_layer, self).__init__()
        self.scaling = nn.ModuleList()
        for i, o in zip(in_channels, out_channels):
            if i == o and transparent:
                self.scaling.append(_transparent())
                continue
            self.scaling.append(_conv2d_norm_relu(i, o, 1))
        print(self.scaling)

    def forward(self, x: list):
        scaled_x = []
        for u, scale in zip(x, self.scaling):
            scaled_x.append(scale(u))
        return scaled_x


class se_gpm2cls_v1(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_out_stages,
        scaling,
        scaling_out_feats,
        global_perception,
        global_perception_out_feats,
        classifer,
        classifer_num_classes
    ):
        super(se_gpm2cls_v1, self).__init__()
        self.backbone = backbone
        self.scale = scaling_layer()
        self.se_gpm = nn.ModuleList()
        self.fc2cls = nn.ModuleList()

    def forward(self, x):
        features = self.backbone(x)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

    """ gpm """
    # x = torch.arange(144).float().view(1, 1, 12, 12).to(device)
    # x = torch.rand((2, 1, 6, 6)).to(device)
    # gpm = global_perception(3, 1).to(device)
    # print(x.size())
    # y = gpm(x)
    # print(y.size())

    """ scaling """
    # x = [
    #     torch.rand((16, 512, 48, 48)).cuda(),
    #     torch.rand((16, 1024, 24, 24)).cuda(),
    #     torch.rand((16, 2048, 12, 12)).cuda()
    # ]
    # scale = scaling_layer(out_channels=[256, 256, 256], transparent=True).cuda()
    # y = scale(x)
    # for o in y: print(o.size())

    """ _resnet """
    # net = _resnet50().cuda()
    # x = torch.rand((16, 3, 448, 448)).cuda()
    # y = net(x)
    # print([a.size() for a in y])


