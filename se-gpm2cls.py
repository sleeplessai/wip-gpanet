import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from fireblast.models.resnet import resnet50, resnet18
except:
    from torchvision.models import resnet50, resnet18


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
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class _dwconv2_norm_leaky_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(_dwconv2_norm_leaky_relu, self).__init__()


class global_perception(nn.Module):
    def __init__(self, n, in_channels, stride=1):
        super(global_perception, self).__init__()
        self.n = n
        self.convs = nn.ModuleList()
        for _ in range(n * n):
            # set groups conv to reduce parameters
            self.convs.append(_conv2d_norm_leaky_relu(n * n * in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels))

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
        for i in range(len(self.convs)):
            t = self.convs[i](x)
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
        b, c, h, w = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class se_gpm_block(nn.Module):
    def __init__(self, in_channels, gpm_n, se_design='se', se_reduction=16):
        super(se_gpm_block, self).__init__()
        assert se_design in ['se', 'pre', 'post', 'identity']
        self.se_design = se_design
        self.se = se_block(in_channels, se_reduction)
        self.gpm = global_perception(gpm_n, in_channels)

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
        return x


class se_gpm2cls(nn.Module):
    def __init__(self, resnet, in_channels=[512, 1024, 2048], reduce2=[64, 64, 64], gpm_n=[2, 2, 2], n_classes=0):
        super(se_gpm2cls, self).__init__()
        self.backbone = resnet

        self.scale = nn.ModuleList()
        self.se_gpm = nn.ModuleList()
        self.fc2cls = nn.ModuleList()

        for i in range(len(in_channels)):
            self.scale.append(_conv2d_norm_leaky_relu(in_channels[i], reduce2[i], 1))
            self.se_gpm.append(se_gpm_block(reduce2[i], gpm_n[i]))
            self.fc2cls.append(nn.Linear(reduce2[i], n_classes, bias=False))

    def forward(self, x):
        stages = self.backbone(x)[-3:]
        y = []
        for i in range(len(self.se_gpm)):
            itmd = self.scale[i](stages[i])
            # print(itmd.size())
            itmd = self.se_gpm[i](itmd)
            # print(itmd.size())
            itmd = F.adaptive_max_pool2d(itmd, 1)
            itmd = torch.flatten(itmd, 1)
            # print(itmd.size())
            itmd = self.fc2cls[i](itmd)
            # print(itmd.size())
            y.append(itmd)
        return y


if __name__ == "__main__":
    device = torch.device('cuda:0')
    x = torch.rand(8, 3, 448, 448).cuda()
    fg_model = se_gpm2cls(
        resnet=_resnet50(),
        reduce2=[512, 512, 512],
        gpm_n=[4, 4, 2],
        n_classes=200
    ).cuda()
    y = fg_model(x)

    torch.save(fg_model.state_dict(), 'se-gpm2cls_resnet50_3x512d.pth')

