import os

from PIL import Image
import torch
import torchvision.transforms.functional as TF

from modeling import gpa2cls_v1_fl2
from fireblast.experiment import set_cuda_visible_devices


if __name__ == "__main__":
    set_cuda_visible_devices(0)
    fl2_model = gpa2cls_v1_fl2('configs/gpa2cls-v1-r50-fl2.yaml', num_classes=200).cuda()

    image_dir = 'cub'
    batch_size = 3
    crop_size = 448
    images = [os.path.join(image_dir, p) for p in os.listdir(image_dir)]
    batch_, batches = [], []
    for idx, img in enumerate(images):
        x = Image.open(img).convert('RGB')
        x = TF.to_tensor(TF.resize(x, (crop_size, crop_size)))
        batch_.append(x)
        if len(batch_) == batch_size or idx == len(images) - 1:
            batches.append(torch.stack(batch_))
            batch_ = []
            # print(batches[-1].size())

    for idx, imgs in enumerate(batches):
        local_imgs = fl2_model(imgs.cuda())
        for i in range(local_imgs.size(0)):
            t = torch.squeeze(local_imgs[i:i + 1]).cpu()
            t = TF.to_pil_image(t)
            # t.show()
            t.save(f'cropped/{idx}-{i}-{local_imgs.size(-1)}x{local_imgs.size(-1)}.jpg')
            print(f'cropped/{idx}-{i}-{local_imgs.size(-1)}x{local_imgs.size(-1)}.jpg saved.')
