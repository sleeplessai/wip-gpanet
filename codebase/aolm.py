from fireblast.models.resnet import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure


class _resnet50(nn.Module):
    def __init__(self, pretrained=True):
        super(_resnet50, self).__init__()
        self.net = resnet50(pretrained=True)
        self.net.fc = None
        self.dropout = nn.Dropout(p=0.5)

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

    def _alom_forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)

        # print(x.size())
        # print('layer4:', self.net.layer4.__len__())
        # print(self.net.layer4[:2])
        conv5_b = self.net.layer4[:2](x)
        x = self.net.layer4[2](conv5_b)

        conv5_c = x
        x = self.net.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        embedding = x

        # conv5_c from last conv layer, \
        # conv5_b is the one in front of conv5_c
        return conv5_c, conv5_b, embedding


def attention_object_location_module(conv5_c, conv5_b):
    A = torch.sum(conv5_c, dim=1, keepdim=True)
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    Mask = (A > a).float()

    A1 = torch.sum(conv5_b, dim=1, keepdim=True)
    a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
    Mask1 = (A1 > a1).float()

    coordinates = []
    for i, m in enumerate(Mask):
        mask_np = m.cpu().numpy().reshape(14, 14)
        component_labels = measure.label(mask_np)

        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas))

        intersection = ((component_labels==(max_idx+1)).astype(int) + (Mask1[i][0].cpu().numpy()==1).astype(int)) == 2
        prop = measure.regionprops(intersection.astype(int))
        if len(prop) == 0:
            bbox = [0, 0, 14, 14]
            print('there is one img no intersection')
        else:
            bbox = prop[0].bbox

        stride = 32
        x_lefttop = bbox[0] * stride - 1
        y_lefttop = bbox[1] * stride - 1
        x_rightlow = bbox[2] * stride - 1
        y_rightlow = bbox[3] * stride - 1
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)

    return coordinates


if __name__ == "__main__":
    net = _resnet50().cuda()
    x = torch.rand((7, 3, 448, 448)).cuda()
    # a, b, c = net._alom_forward(x)
    # print(a.size(), b.size(), c.size())
    a = torch.rand((1, 3, 14, 14)).cuda()
    b = torch.rand((1, 3, 14, 14)).cuda()
    coor = attention_object_location_module(a, b)
    print(coor)
