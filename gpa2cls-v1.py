import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from fireblast.models.resnet import resnet50, resnet18
except:
    from torchvision.models import resnet50, resnet18
import numpy as np
from skimage import measure


### Utility modules ###

class _backbone(nn.Module):
    def __init__(self, pretrained=True, pretrained_file=None, num_classes=0):
        super(_backbone, self).__init__()
        if not pretrained_file:
            self.net = resnet50(pretrained=True)
        elif pretrained_file and num_classes:
            print(f'Load weights from {pretrained_file}')
            self.net = resnet50(pretrained=False)
            self.net.fc = nn.Linear(2048, num_classes)
            self.net.load_state_dict(torch.load(pretrained_file))
        self.dropout5 = nn.Dropout(p=0.5)

    @property
    def out_channels_per_stage(self):
        return [64, 256, 512, 1024, 2048]

    def forward(self, x, scda_stage=-1, multistage=False):
        scda = [None]

        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x1 = self.net.maxpool(x)             # stage 1

        conv2_b = self.net.layer1[:-1](x1)
        x2 = self.net.layer1[-1](conv2_b)    # stage 2
        conv2_c = x2
        scda.append((conv2_c, conv2_b))

        conv3_b = self.net.layer2[:-1](x2)
        x3 = self.net.layer2[-1](conv3_b)    # stage 3
        conv3_c = x3
        scda.append((conv3_c, conv3_b))

        conv4_b = self.net.layer3[:-1](x3)
        x4 = self.net.layer3[-1](conv4_b)    # stage 4
        conv4_c = x4
        scda.append((conv4_c, conv4_b))

        conv5_b = self.net.layer4[:-1](x4)
        x5 = self.net.layer4[-1](conv5_b)    # stage 5
        conv5_c = x5
        scda.append((conv5_c, conv5_b))

        x = self.net.avgpool(x5)
        x = x.view(x.size(0), -1)
        x = self.dropout5(x)
        repr_x5 = x

        conv_c, conv_b = scda[scda_stage]
        # conv5_c from last conv layer, \
        # conv5_b is the one in front of conv5_c
        if not multistage: return conv_c, conv_b, repr_x5
        return [x1, x2, x3, x4, x5]


def _transparent():
    return nn.Identity()


def _dense_layer(in_features, out_features, bias=False):
    return nn.Linear(in_features, out_features, bias)


def _dropout(p=0.5):
    return nn.Dropout(p=p, inplace=True)


class _conv2d_norm_relu(nn.Module):
    _negative_slope = 0.5 / np.e     # 0.18393972058572117

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


### Basic components/blocks for GPA2Cls-v1 ###

class focal_locator:
    def __init__(self, size: int = 448, stride: int = 32, focal_size: int = 384):
        self.size = size
        self.stride = stride
        self.focal_size = focal_size

    def _attention_object_location_module(self, conv5_c, conv5_b, image_wh=448, stride=32):
        A = torch.sum(conv5_c, dim=1, keepdim=True)
        a = torch.mean(A, dim=[2, 3], keepdim=True) * 1.0
        mask_c = (A > a).float()
        A_ = torch.sum(conv5_b, dim=1, keepdim=True)
        a_ = torch.mean(A_, dim=[2, 3], keepdim=True) * 1.0
        mask_b = (A_ > a_).float()

        boxes = []
        for i, mask in enumerate(mask_c):
            mask = mask.cpu().numpy().reshape(image_wh // stride, image_wh // stride)
            component_labels = measure.label(mask)
            properties = measure.regionprops(component_labels)
            areas = []
            for prop in properties:
                areas.append(prop.area)
            max_idx = areas.index(max(areas))
            intersection = ((component_labels == (max_idx + 1)).astype(int) +
                            (mask_b[i][0].cpu().numpy() == 1).astype(int)) == 2
            prop = measure.regionprops(intersection.astype(int))
            if len(prop) == 0:
                bbox = [0, 0, image_wh // stride, image_wh // stride]
            else:
                bbox = prop[0].bbox
            upperleft_x, upperleft_y = bbox[0] * stride - 1, bbox[1] * stride - 1
            lowerright_x, lowerright_y = bbox[2] * stride - 1, bbox[3] * stride - 1
            upperleft_x = 0 if upperleft_x < 0 else upperleft_x
            upperleft_y = 0 if upperleft_y < 0 else upperleft_y
            boxes.append([upperleft_x, upperleft_y, lowerright_x, lowerright_y])
        return boxes

    def locate(self, image_batch, conv_c, conv_b):
        boxes = self._attention_object_location_module(conv_c.detach(), conv_b.detach(), self.size, self.stride)
        focus = torch.zeros((image_batch.size(0), 3, self.focal_size, self.focal_size)).cuda()
        for i in range(image_batch.size(0)):
            ulx, uly, lrx, lry = boxes[i]
            focus[i:i + 1] = F.interpolate(image_batch[i:i + 1, :, ulx:lrx + 1, uly:lry + 1],
                                           size=(self.focal_size, self.focal_size), mode='bilinear', align_corners=True)
        return focus


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


class gpa_block(nn.Module):
    def __init__(self, in_channels, n, se_design='se', se_reduction=16):
        super(gpa_block, self).__init__()
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


### Main layers for GPA2Cls-v1 ###

class scaling_layer(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], out_channels=[512, 512, 512], transparent=False):
        assert len(in_channels) == len(out_channels)
        super(scaling_layer, self).__init__()
        self.scaling = nn.ModuleList()
        for i, o in zip(in_channels, out_channels):
            if i == o and transparent:
                self.scaling.append(_transparent())
                continue
            self.scaling.append(_conv2d_norm_relu(i, o, 1))
        # print(self.scaling)

    def forward(self, x: list):
        scaled_x = [scale(u) for u, scale in zip(x, self.scaling)]
        return scaled_x


class gpa_layer(nn.Module):
    def __init__(self, in_channels=[512, 512, 512], split_sizes=[2, 2, 2], se_designs=['se', 'se', 'se'], transparent=True):
        assert len(in_channels) == len(split_sizes) == len(se_designs)
        super(gpa_layer, self).__init__()
        self.gpa = nn.ModuleList()
        for i, n, d in zip(in_channels, split_sizes, se_designs):
            if n == 1 and transparent:
                self.gpa.append(_transparent())
                continue
            self.gpa.append(gpa_block(i, n, d))
        # print(self.gpa)

    def forward(self, x):
        global_x = [gpm(u) for u, gpm in zip(x, self.gpa)]
        return global_x


class classifier(nn.Module):
    def __init__(self, in_features=[512, 512, 512], num_classes=0, pooling=F.adaptive_max_pool2d, dropout=True):
        super(classifier, self).__init__()
        self.dropout = nn.ModuleList() if dropout else None
        self.fc = nn.ModuleList()
        for i in in_features:
            self.fc.append(_dense_layer(i, num_classes))
            if dropout:
                self.dropout.append(_dropout())
        self.pooling = pooling
        # print(self.dropout)
        # print(self.fc)

    def forward(self, x):
        repr_x = [self.pooling(o, 1).view(o.size(0), -1) for o in x]    # pooling and flatten
        if self.dropout:
            repr_x = [drop(o) for o, drop in zip(repr_x, self.dropout)]
        logits = [clas(e) for e, clas in zip(repr_x, self.fc)]
        return logits


### TODO: GPA2Cls-v1 definition ###

class gpa2cls_v1(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module = None,
        locator: focal_locator = None,
        scaling: nn.Module = None,
        global_per_attn: nn.Module = None,
        classifier: nn.Module = None
    ):
        super(gpa2cls_v1, self).__init__()
        self.backbone = backbone
        self.scaling = scaling
        self.global_per_attn = global_per_attn

    def forward(self, x_batch):
        x = self.backbone(x_batch)
        x = self.scaling(x)
        x = self.global_per_attn(x)


### Components tests ###

def _component_ready():
    print(""" *single gpm """)
    x = torch.arange(144).float().view(1, 1, 12, 12).to(device)
    gpm = global_perception(3, 1).to(device)
    print(x.size())
    y = gpm(x)
    print(y.size())
    print()

    print(""" *backbone """)
    net = _backbone().cuda()
    print(net.out_channels_per_stage)

    x = torch.rand((16, 3, 448, 448)).cuda()
    a, b, c = net.forward(x)
    print(a.size(), b.size(), c.size())

    x = torch.rand((16, 3, 384, 384)).cuda()
    y = net.forward(x, multistage=True)
    print([a.size() for a in y])
    print()

    print(""" *focal_locator """)
    x = torch.rand((16, 3, 448, 448)).cuda()
    x1 = torch.rand((16, 2048, 14, 14)).cuda()
    x2 = torch.rand((16, 2048, 14, 14)).cuda()
    fl = focal_locator()
    b = fl.locate(x, x1, x2)
    b.requires_grad = True
    print(b.size(), b.requires_grad)
    print()

    print(""" *scaling """)
    x = [
        torch.rand((16, 512, 48, 48)).cuda(),
        # torch.rand((16, 1024, 24, 24)).cuda(),
        torch.rand((16, 2048, 12, 12)).cuda()
    ]
    scale = scaling_layer(in_channels=[512, 2048] ,out_channels=[512, 512], transparent=True).cuda()
    print(scale)
    x = scale(x)
    for o in x: print(o.size())
    print()

    print(""" *global_per_attn layer """)
    gpa = gpa_layer(in_channels=[512, 512], split_sizes=[4, 1], se_designs=['se', 'se']).cuda()
    print(gpa)
    x = gpa(x)
    print()

    print(""" *fully-connected layer """)
    for o in x: print(o.size())
    clz = classifier(in_features=[512, 512], num_classes=196, dropout=True).cuda()
    print(clz)
    x = clz(x)
    for o in x: print(o.size())
    print()



if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

    _component_ready()
