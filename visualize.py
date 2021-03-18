
import os

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from gpa2cls import gpa2cls_v1, _backbone
from fireblast import set_cuda_visible_devices


class resnet_grad_cam(_backbone):
    def __init__(self, model_type, pretrained, pretrained_file, num_classes):
        super(resnet_grad_cam, self).__init__(model_type=model_type, pretrained=pretrained, pretrained_file=pretrained_file, num_classes=num_classes)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x5 = x

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return x5, x

class gpa2cls_v1a_grad_cam(gpa2cls_v1):
    def __init__(self, cfg_file, num_classes):
        super(gpa2cls_v1a_grad_cam, self).__init__(cfg_file, num_classes)

    def forward(self, x_batch):
        x5c, x5b, x5 = self.backbone(x_batch)
        x_focal = self.locator.locate(x_batch, x5c, x5b)

        x_feats = np.array(self.backbone(x_focal, multistage=True))[self.stages].tolist()
        x_feats = self.scaling(x_feats)
        for i in range(len(x_feats) - 1):
            x_feats[i] = F.adaptive_max_pool2d(x_feats[i], x_feats[-1].size(-1))
        x_feats = torch.cat(x_feats, dim=1)
        x_feats = self.gp_attn([x_feats])[0]

        return x_focal, x_feats, self.clf([x5, x_feats])[1]


def visulaize_grad_cam(model, src, dst,  tr=None, norm=False, eta=0.5, plot=False, idx=None):
    '''
    Visualize class activation map
    https://blog.csdn.net/sinat_37532065/article/details/103362517
    :param model: weight-ready pytorch model
    :param src: source image path
    :param dst: destination image path
    :param tr: torchvision transfomation
    :param plot: matplotlib show
    :return: none
    '''
    # set model type flag
    if isinstance(model, gpa2cls_v1a_grad_cam):
        flag = 'gpa2cls'
    elif isinstance(model, resnet_grad_cam):
        flag = 'resnet'

    # laod and preprocess
    img = Image.open(src).convert("RGB")
    if tr: img = tr(img)
    T.functional.to_pil_image(img.squeeze()).save(f"vis/{idx}o.jpg")
    if norm: T.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img = img.unsqueeze(0).cuda()

    # model forward to get features and ouput
    model = model.cuda()
    model.eval()
    if flag == 'gpa2cls':
        focal, feats, output = model.forward(img)
        focal = T.functional.to_pil_image(focal.squeeze()).save(f"vis/{idx}t.jpg")
    elif flag == 'resnet':
        feats, output = model.forward(img)

    feat_dims = feats.size(1)

    # extract hook to get grad
    def extract(g):
        global features_grad
        features_grad = g

    # get the class with highest prediction score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    feats.register_hook(extract)
    pred_class.backward() # calculate grad

    grads = features_grad   # store grad
    pooled_grads = F.adaptive_avg_pool2d(grads, (1, 1))

    # batch_size = 1
    pooled_grads = pooled_grads[0]
    feats = feats[0]
    # feat_dims is channel # of feature
    for i in range(feat_dims):
        feats[i, ...] *= pooled_grads[i, ...]

    # same as the keras impl
    heat_map = feats.detach().cpu().numpy()
    heat_map = np.mean(heat_map, axis=0)

    heat_map = np.maximum(heat_map, 0)
    heat_map /= np.max(heat_map)

    # plot heat map
    if plot:
        plt.matshow(heat_map)
        plt.axis("off")
        plt.show()

    # cv2 load original image
    if flag == 'gpa2cls':
        img = cv2.imread(f"vis/{idx}t.jpg")
    elif flag == 'resnet':
        img = cv2.imread(f"vis/{idx}o.jpg")
    heat_map = cv2.resize(heat_map, (img.shape[1], img.shape[0]))  # resize heat map to original size
    heat_map = np.uint8(255. * heat_map)  # heat map to rgb mode
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)  # apply heat map to the orginal
    superimposed_img = heat_map * eta + img  # eta is intensity ratio
    cv2.imwrite(dst, superimposed_img)  # save cam to local dst


if __name__ == "__main__":
    set_cuda_visible_devices(0)

    cfg1 = 100
    ptw1 = "saved/resnet50-FGVC-Aircraft.pth"
    # ptw1 = "saved/resnet50-CUB-200.pth"
    # ptw1 = "saved/resnet50-Cars196.pth"
    net1 = resnet_grad_cam(model_type="r50", pretrained=False, pretrained_file=ptw1, num_classes=cfg1)

    cfg2 = "configs/gpa2cls-v1-x50-ms35-2560d2x2.yaml"
    ptw2 = "saved/gpa2cls-v1-x50-ms35-2560d2x2-Air.pth"
    net2 = gpa2cls_v1a_grad_cam(cfg2, cfg1)
    net2.load_state_dict(torch.load(ptw2))

    # # torch.Size([1, 2560, 12, 12]) torch.Size([1, 200])
    tr = T.Compose([T.Resize((512, 512)), T.RandomCrop((448, 448)), T.ToTensor()])

    # cub visualization
    for idx in range(1, 61):
        visulaize_grad_cam(net1, src=f"ds_samples/cub/{idx}.jpg", dst=f"vis/{idx}r.jpg", tr=tr, norm=True, eta=0.6, idx=idx)
        visulaize_grad_cam(net2, src=f"ds_samples/cub/{idx}.jpg", dst=f"vis/{idx}g.jpg", tr=tr, norm=True, eta=0.6, idx=idx)

    # car visualization
    for idx in range(1, 61):
        visulaize_grad_cam(net1, src=f"ds_samples/car/{idx}.jpg", dst=f"vis/{idx}r.jpg", tr=tr, norm=True, eta=0.6, idx=idx)
        visulaize_grad_cam(net2, src=f"ds_samples/car/{idx}.jpg", dst=f"vis/{idx}g.jpg", tr=tr, norm=True, eta=0.6, idx=idx)

    # air visualization
    for idx in range(1, 61):
        visulaize_grad_cam(net1, src=f"ds_samples/air/{idx}.jpg", dst=f"vis/{idx}r.jpg", tr=tr, norm=True, eta=0.6, idx=idx)
        visulaize_grad_cam(net2, src=f"ds_samples/air/{idx}.jpg", dst=f"vis/{idx}g.jpg", tr=tr, norm=True, eta=0.6, idx=idx)

