import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from fireblast.models.resnet import resnet50, resnet18
except:
    from torchvision.models import resnet50, resnet18
import numpy as np
import random


def jigsaw_generator(features, n):
    l = [[a, b] for a in range(n) for b in range(n)]
    block_size = features.size(-1) // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = features.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws


class zigzag(nn.Module):
    def __init__(self, pretrained=True, class_cnt=0):
        self._id = 'zigzagjig-0riginal'
        super(zigzag, self).__init__()
        self.net = resnet50(pretrained=True)
        self.net.fc = None
        self.net.avgpool = None
        self.jiggen = jigsaw_generator
        self.fc = nn.Linear(2048, class_cnt, bias=False)

    def forward(self, x, feats=False):  # todo: partial_rate arg
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        # stage-1
        s1 = self.net.maxpool(x)
        # stage-2
        s2 = self.net.layer1(s1)
        # print(s2.size())
        p1, p2 = torch.split(s2, s2.size(1)//2, dim=1)
        # print(p1.size() == p2.size())
        p1 = self.jiggen(p1, 8)
        s2 = torch.cat((p1, p2), dim=1)
        # stage-3
        s3 = self.net.layer2(s2)
        # print(s3.size())
        p1, p2 = torch.split(s3, s3.size(1)//2, dim=1)
        # print(p1.size() == p2.size())
        p1 = self.jiggen(p1, 4)
        s3 = torch.cat((p1, p2), dim=1)
        # stage-4
        s4 = self.net.layer3(s3)
        # print(s4.size())
        p1, p2 = torch.split(s4, s4.size(1)//2, dim=1)
        # print(p1.size() == p2.size())
        p1 = self.jiggen(p1, 2)
        s4 = torch.cat((p1, p2), dim=1)
        # stage-5
        s5 = self.net.layer4(s4)
        # avgpool, fc
        x = F.adaptive_avg_pool2d(s5, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if feats:
            return [s1, s2, s3, s4, s5], x
        else:
            return x


if __name__ == "__main__":
    net = zigzag(class_cnt=200).cuda()
    x = torch.rand((16, 3, 448, 448)).cuda()
    import time
    t = time.time()
    # y, c = net(x)
    y = net(x)
    print(time.time() - t)
    print(y.size())
