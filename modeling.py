import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpa2cls import gpa2cls_v1


class gpa2cls_v1a(gpa2cls_v1):
    def __init__(self, cfg_file, num_classes):
        super(gpa2cls_v1a, self).__init__(cfg_file, num_classes)

    def forward(self, x_batch):
        x5c, x5b, x5 = self.backbone(x_batch)
        x_focal = self.locator.locate(x_batch, x5c, x5b)

        x_feats = np.array(self.backbone(x_focal, multistage=True))[self.stages].tolist()
        x_feats = self.scaling(x_feats)
        for i in range(len(x_feats) - 1):
            x_feats[i] = F.adaptive_max_pool2d(x_feats[i], x_feats[-1].size(-1))
        x_feats = torch.cat(x_feats, dim=1)
        x_feats = self.gp_attn([x_feats])[0]

        return self.clf([x5, x_feats])


class gpa2cls_v1_fl2(gpa2cls_v1):
    def __init__(self, cfg_file, num_classes):
        super(gpa2cls_v1_fl2, self).__init__(cfg_file, num_classes)

    def forward(self, x_batch):
        x5c, x5b, _ = self.backbone(x_batch)
        x_focal = self.locator.locate2(x_batch, x5c, x5b)
        return x_focal


if __name__ == "__main__":
    """ *fl1 models """
    # net = gpa2cls_v1a('configs/gpa2cls-v1-r50-ms35-1024d2x2.yaml', num_classes=200)
    # net = gpa2cls_v1a('configs/gpa2cls-v1-r50-ms35-2560d2x2.yaml', num_classes=200)
    # net = gpa2cls_v1a('configs/gpa2cls-v1-r50-ms345-3072d2x2.yaml', num_classes=200)
    # net = gpa2cls_v1a('configs/gpa2cls-v1-r50-ms45-3072d2x2.yaml', num_classes=200)
    net = gpa2cls_v1a('configs/gpa2cls-v1-x50-ms35-2560d3x3.yaml', num_classes=200)

    """ *fl2 models """
    # net = gpa2cls_v1_fl2('configs/gpa2cls-v1-r50-fl2.yaml', num_classes=200)    # fl2
    net = gpa2cls_v1a('configs/gpa2cls-v1-r50-fl2-ms35-2560d2x2.yaml', num_classes=200)

    print(net)
    print(net.model_id)
    print(net.cfg_node)

    exit(0)
    net = net.cuda()
    # torch.save(net.state_dict(), 'gpa2clsv1-null.pth')
    x = torch.rand((8, 3, 448, 448)).cuda()
    y = net(x)
    # print(y[0].size(), y[1].size())

