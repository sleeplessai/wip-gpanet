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


class gpa2cls_v1b(gpa2cls_v1):
    def __init__(self, cfg_file, num_classes):
        super(gpa2cls_v1b, self).__init__(cfg_file, num_classes)

    def forward(self, x_batch):
        x5c, x5b, x5 = self.backbone(x_batch)
        x_focal = self.locator.locate(x_batch, x5c, x5b)

        x_feats = np.array(self.backbone(x_focal, multistage=True))[self.stages].tolist()
        x_feats = self.scaling(x_feats)
        x_global = self.gp_attn(x_feats)
        x_global = [F.adaptive_max_pool2d(g, 1).view(g.size(0), -1) for g in x_global]
        xg = torch.cat(x_global, dim=1)
        x5 = F.adaptive_avg_pool2d(x5, 1).view(x5.size(0), -1)

        return self.clf([x5, xg])


if __name__ == "__main__":
    # net = gpa2cls_v1a('configs/gpa2cls-v1-r50-1024d2x2.yaml', num_classes=200).cuda()
    # net = gpa2cls_v1b('configs/gpa2cls-v1-r50-mg442-1536d.yaml', num_classes=200).cuda()
    net = gpa2cls_v1a('configs/gpa2cls-v1-r50-2560d2x2.yaml', num_classes=200).cuda()

    print(net.cfg_node)
    print(net.model_id)
    torch.save(net.state_dict(), 'gpa2clsv1-null.pth')
    x = torch.rand((12, 3, 448, 448)).cuda()
    y = net(x)

