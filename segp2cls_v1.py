import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from fireblast.models.resnet import resnet50, resnet18
except:
    from torchvision.models import resnet50, resnet18
import numpy as np


class _transparent(nn.Module):
    def __init__(self): super(_transparent, self).__init__()
    def forward(self, x): return x


class _resnet50(nn.Module):
    def __init__(self, pretrained=True):
        super(_resnet50, self).__init__()
        self.net = resnet50(pretrained=True)
        self.net.fc = None
        self.net.avgpool = None

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


class _conv2d_norm_leaky_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(_conv2d_norm_leaky_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=.5/np.e, inplace=True)

    def forward(self, x, activate=True):
        x = self.conv(x)
        x = self.norm(x)
        if activate: x = self.relu(x)
        return x


class _se_quarter_bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, se_design='se', se_reduction=16):
        assert se_design in ['se', 'pre', 'post', 'identity', None]
        super(_se_quarter_bottleneck, self).__init__()

        quarter = in_channels // 4
        self.conv1 = _conv2d_norm_leaky_relu(in_channels, quarter, 1, groups=quarter)
        self.conv2 = _conv2d_norm_leaky_relu(quarter, quarter, 3, padding=1, groups=quarter)
        self.conv3 = _conv2d_norm_leaky_relu(quarter, out_channels, 1, groups=out_channels)
        if in_channels != out_channels:
            self.down = _conv2d_norm_leaky_relu(in_channels, out_channels, 1, groups=out_channels)
        self.se_design = se_design

        if not se_design:
            self.se = None
        elif se_design in ['se', 'post']:
            self.se = se_block(out_channels, se_reduction)
        elif se_design in ['pre', 'identity']:
            self.se = se_block(in_channels, se_reduction)

    def forward(self, x, activate=False):
        print(x.size())
        identity = x
        if not self.se_design:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x, activate=False)
            x += identity
        elif self.se_design == 'se':
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            se = self.se(x)
            x += identity
        elif self.se_design == 'pre':
            x = self.se(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x, activate=False)
            x += identity
        elif self.se_design == 'post':
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x, activate=False)
            x += identity
            x = F.leaky_relu(x, negative_slope=.5/np.e)
            x = self.se(x)
        elif self.se_design == 'identity':
            print(x.size())
            print(identity.size())
            print(self.conv1)
            x = self.conv1(identity)
            x = self.conv2(x)
            x = self.conv3(x)
            x += self.se(identity)
        x = F.leaky_relu(x, negative_slope=.5/np.e)
        return x


class se_global_perception(nn.Module):
    def __init__(self, in_channels, n, se_design='se'):
        super(se_global_perception, self).__init__()
        self.n = n
        self.recons = nn.ModuleList()
        for _ in range(n * n):
            # se-gpm2cls use se-bottleneck as recon conv
            self.recons.append(_se_quarter_bottleneck(n * n * in_channels, in_channels, se_design=se_design, se_reduction=16))

    def forward(self, x):
        # print(x.size())
        spx = torch.chunk(x, self.n, -2)
        spxx = []
        for a in spx:
            spx1 = torch.chunk(a, self.n, -1)
            for b in spx1:
                spxx.append(b)
                # print(b.size())
        x = torch.cat(spxx, 1)
        # print(x.size())
        catx = []
        restore = []
        for i in range(len(self.recons)):
            t = self.recons[i](x)
            # print(t.size())
            catx.append(t)
            if len(catx) == self.n:
                a = torch.cat(catx, -1)
                # print(a.size())
                catx = []
                restore.append(a)
            # print(t.size())
        x = torch.cat(restore, -2)
        # print(x.size())
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


class se_gpm2cls_v1(nn.Module):
    def __init__(self, resnet, in_channels=[512, 1024, 2048],
                 reduce2=[64, 64, 64], gpm_split_sz=[2, 2, 2], se_design='se', n_classes=0):
        super(se_gpm2cls_v1, self).__init__()
        self.backbone = resnet

        self.scale = nn.ModuleList()
        self.se_gpm = nn.ModuleList()
        self.fc2cls = nn.ModuleList()

        for i in range(len(in_channels)):
            if in_channels[i] == reduce2[i]:
                self.scale.append(_transparent())
            else:
                self.scale.append(_conv2d_norm_leaky_relu(in_channels[i], reduce2[i], 1))
            self.se_gpm.append(se_global_perception(reduce2[i], gpm_split_sz[i], se_design=se_design))
            if i == 0: continue
            self.fc2cls.append(nn.Linear(reduce2[i] + reduce2[0], n_classes, bias=False))
        print(self.fc2cls)


    def forward(self, x):
        stages = self.backbone(x)[-3:]
        y = []
        for i in range(len(self.se_gpm)):
            itmd = self.scale[i](stages[i])
            itmd = self.se_gpm[i](itmd)
            y.append(itmd)
        z = y.pop(0)    # stage3
        for i in range(len(y)):
            z = F.adaptive_max_pool2d(z, y[i].size()[-1])
            y[i] = torch.cat((y[i], z), dim=1)
            y[i] = F.adaptive_avg_pool2d(y[i], 1)
            y[i] = torch.flatten(y[i], 1)
            y[i] = self.fc2cls[i](y[i])
        return y


def to_text(model, model_id):
    with open(f'{model_id}.txt', 'w') as f:
        f.write(str(model))


if __name__ == "__main__":
    device = torch.device('cuda:0')
    x = torch.rand(4, 3, 448, 448).to(device)
    fg_model = se_gpm2cls_v1(
        resnet=_resnet50(),
        reduce2=[512, 1024, 1024],
        gpm_split_sz=[4, 4, 2],
        se_design='identity',
        n_classes=200
    ).to(device)
    y = fg_model(x)
    for u in y: print(u.size())

    model_id = 'se-gpm2cls-v1_resnet50'

    torch.save(fg_model.state_dict(), f'{model_id}.pth')
    to_text(fg_model, f'{model_id}')

