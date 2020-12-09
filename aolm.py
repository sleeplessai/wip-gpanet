# try:
# from fireblast.models.resnet import resnet50
# except:
from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure


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

    def _aolm_forward(self, x):
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


def attention_object_location_module(conv5_c, conv5_b, image_wh=448, stride=32):
    A = torch.sum(conv5_c, dim=1, keepdim=True)
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    mask_c = (A > a).float()

    A_ = torch.sum(conv5_b, dim=1, keepdim=True)
    a_ = torch.mean(A_, dim=[2, 3], keepdim=True)
    mask_b = (A_ > a_).float()

    coordinates = []
    for i, m in enumerate(mask_c):
        mask_np = m.cpu().numpy().reshape(14, 14)
        component_labels = measure.label(mask_np)

        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas))

        intersection = ((component_labels==(max_idx+1)).astype(int) + (mask_b[i][0].cpu().numpy()==1).astype(int)) == 2
        prop = measure.regionprops(intersection.astype(int))
        if len(prop) == 0:
            bbox = [0, 0, image_wh // stride, image_wh // stride]
            print('there is one img no intersection')
        else:
            bbox = prop[0].bbox

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

    net = _resnet50(pth_path=r"saved/resnet50-CUB-200.pth").cuda()

    # exit(0)
    from PIL import Image
    import cv2
    from torchvision.transforms.functional import to_tensor, resize, to_pil_image, crop
    import os

    image_dir = 'cub'
    for idx, im in enumerate([os.path.join(image_dir, p) for p in os.listdir(image_dir)]):
        x = Image.open(im).convert('RGB')
        x = torch.unsqueeze(to_tensor(resize(x, (448, 448))), dim=0).float().cuda()
        if x.size(1) == 1: x = torch.cat((x, x, x), dim=1)
        conv_c, conv_b, _ = net._aolm_forward(x)
        ulx, uly, lrx, lry = attention_object_location_module(conv_c, conv_b)[0]
        print(ulx, uly, lrx, lry, idx)

        x = to_pil_image(torch.squeeze(x).cpu())
        x = crop(x, ulx, uly, lrx - ulx, lry - uly)
        # print(x.size())
        w = lrx - ulx
        h = lry - uly
        ratio = w / h
        # todo: keep ratio 384
        x = resize(x, (w, h), interpolation=Image.ANTIALIAS)
        x.save(f'cropped/cropped-{idx}.jpg')
        # x.show()
