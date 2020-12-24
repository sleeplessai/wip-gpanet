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

    def _aolm_forward(self, x, scda_stage=-1):
        scda = [None]

        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)     # s1

        conv2_b = self.net.layer1[:-1](x)
        x = self.net.layer1[-1](conv2_b)
        conv2_c = x
        scda.append((conv2_c, conv2_b))    # s2

        conv3_b = self.net.layer2[:-1](x)
        x = self.net.layer2[-1](conv3_b)
        conv3_c = x
        scda.append((conv3_c, conv3_b))    # s3

        conv4_b = self.net.layer3[:-1](x)
        x = self.net.layer3[-1](conv4_b)
        conv4_c = x
        scda.append((conv4_c, conv4_b))    # s4

        conv5_b = self.net.layer4[:2](x)
        x = self.net.layer4[2](conv5_b)
        conv5_c = x
        scda.append((conv5_c, conv5_b))    # s5

        x = self.net.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        embedding = x

        conv_c, conv_b = scda[scda_stage]
        # conv5_c from last conv layer, \
        # conv5_b is the one in front of conv5_c
        return conv_c, conv_b, embedding


def attention_object_location_module(conv5_c, conv5_b, image_wh=448, stride=32):
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
            print("There is an image no mask intersection")
        else:
            bbox = prop[0].bbox

        upperleft_x, upperleft_y = bbox[0] * stride - 1, bbox[1] * stride - 1
        lowerright_x, lowerright_y = bbox[2] * stride - 1, bbox[3] * stride - 1
        upperleft_x = 0 if upperleft_x < 0 else upperleft_x
        upperleft_y = 0 if upperleft_y < 0 else upperleft_y

        boxes.append([upperleft_x, upperleft_y, lowerright_x, lowerright_y])

    return boxes


def iterative_aolm(
    pth_path: str = "saved/resnet50-CUB-200.pth",
    image_dir: str = "cub",
    crop_size: int = 448,
    focus_size: int = 224
) -> None:

    from PIL import Image
    from torchvision.transforms.functional import to_tensor, resize, to_pil_image, crop
    import os

    net = _resnet50(pth_path=pth_path).cuda()
    images = [os.path.join(image_dir, p) for p in os.listdir(image_dir)]

    for idx, im in enumerate(images):
        x = Image.open(im).convert('RGB')
        x = torch.unsqueeze(to_tensor(resize(x, (crop_size, crop_size))), dim=0).float().cuda()
        if x.size(1) == 1: x = torch.cat((x, x, x), dim=1)

        conv_c, conv_b, _ = net._aolm_forward(x, scda_stage=-1)
        ulx, uly, lrx, lry = attention_object_location_module(conv_c, conv_b, image_wh=crop_size, stride=32)[0]
        focus_w, focus_h = lrx - ulx, lry - uly
        # print(idx, ':', ulx, uly, lrx, lry)
        x = to_pil_image(torch.squeeze(x).cpu())
        x = crop(x, ulx, uly, focus_w, focus_h)
        # ratio = focus_w / focus_h

        # if ratio > 1.:  # keep ratio
        #     focus_w = focus_size
        #     focus_h = int(focus_size / ratio)
        # elif ratio < 1.:
        #     focus_h = focus_size
        #     focus_w = int(focus_size * ratio)
        # else:
        #     focus_w, focus_h = focus_size, focus_size
        x = resize(x, (focus_w, focus_h), interpolation=Image.BILINEAR)
        # x.save(f'cropped/{idx}_.jpg')
        # x.show()


def tensorized_aolm(
    pth_path: str = "saved/resnet50-CUB-200.pth",
    image_dir: str = "cub",
    crop_size: int = 448,
    focus_size: int = 224,
    batch_size: int = 12
) -> None:

    from PIL import Image
    from torchvision.transforms.functional import to_tensor, resize, to_pil_image, crop
    import os

    images = [os.path.join(image_dir, p) for p in os.listdir(image_dir)]
    batch_, batches = [], []
    for idx, img in enumerate(images):
        x = Image.open(img).convert('RGB')
        x = to_tensor(resize(x, (crop_size, crop_size)))
        batch_.append(x)
        if len(batch_) == batch_size or idx == len(images) - 1:
            batches.append(torch.stack(batch_))
            batch_ = []
            # print(batches[-1].size())

    net = _resnet50(pth_path=pth_path).cuda()
    boxes_, boxes = [], []
    for idx, imgs in enumerate(batches):
        imgs = imgs.cuda()
        conv_c, conv_b, _ = net._aolm_forward(imgs, scda_stage=-1)
        boxes_ = attention_object_location_module(conv_c, conv_b, image_wh=crop_size, stride=32)
        boxes += boxes_
        local_imgs = torch.zeros((imgs.size(0), 3, focus_size, focus_size)).cuda()
        for i in range(imgs.size(0)):
            ulx, uly, lrx, lry = boxes_[i]
            local_imgs[i:i + 1] = F.interpolate(imgs[i:i + 1, :, ulx:lrx + 1, uly:lry + 1],
                                                size=(focus_size, focus_size), mode='bilinear', align_corners=True)
            t = torch.squeeze(local_imgs[i:i + 1]).cpu()
            t = to_pil_image(t)
            # t.show()
            t.save(f'cropped/{idx * batch_size + i}_.jpg')

    return boxes

if __name__ == "__main__":
    # iterative_aolm()
    tensorized_aolm(focus_size=384)
