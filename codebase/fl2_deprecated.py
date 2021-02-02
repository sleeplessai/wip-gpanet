import os

from PIL import Image
import torch
import torchvision.transforms.functional as TF

from modeling import gpa2cls_v1_fl2
from fireblast.experiment import set_cuda_visible_devices

def deprecated_locate2(self, image_batch, conv_c, conv_b):
    assert isinstance(self.focal_size, list) and len(self.focal_size) >= 2
    boxes = self._get_bbox(conv_c.detach(), conv_b.detach(), self.size, self.stride)
    _fsz_idx = torch.multinomial(torch.ones_like(torch.tensor(self.focal_size)).float(), 1)
    focal_size = self.focal_size[_fsz_idx]
    focus = torch.zeros((image_batch.size(0), 3, focal_size, focal_size)).cuda()
    for i in range(image_batch.size(0)):
        ulx, uly, lrx, lry = boxes[i]
        focus[i:i + 1] = TF.to_tensor(
            TF.resize(TF.to_pil_image(image_batch[i:i + 1, :, ulx:lrx + 1, uly:lry + 1].squeeze()),
                      size=(focal_size, focal_size), interpolation=Image.LANCZOS)
        )
        # TODO: Cut-and-Paste
        # focus[i:i + 1, :, focal_size - (lrx - ulx):focal_size, focal_size - (lry - uly):focal_size] = image_batch[i:i + 1, :, ulx:lrx + 1, uly:lry + 1]
        # focus[i:i + 1, :, 0:lrx - ulx + 1, 0:lry - uly + 1] = image_batch[i:i + 1, :, ulx:lrx + 1, uly:lry + 1]
    return focus


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
