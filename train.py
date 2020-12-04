import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from fireblast.experiment.default import Experiment
from fireblast.experiment.default import default_cub200, default_cars196, default_aircraft
from fireblast.experiment.loop import Loop
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
  print(torch.backends.cudnn.enabled)
  torch.cuda.empty_cache()

  device_id = 'cuda'
  epoch_cnt = 60
  expt = Experiment()
  # default_cub200(expt, data_loc='/home/mlss/data/CUB_200_2011', loader=True)
  # default_cars196(expt, data_loc='/home/mlss/data/cars196', loader=True)
  default_aircraft(expt, data_loc='/home/mlss/data/fgvc-aircraft-2013b', trainval=True, loader=True)

  from ablation import ABLATION_MODELS
  from segp2cls_v0 import _resnet50

  model = ABLATION_MODELS['0c'](
    resnet=_resnet50(),
    n_classes=expt.category_cnt
  )
  model_id = model._id

  # optimizer
  sgdm_lr = np.pi / np.e / 100.  # 0.0115~
  sgdm_lrs = [sgdm_lr * .1, sgdm_lr, sgdm_lr, sgdm_lr, sgdm_lr]
  schd_t, schd_d, schd_r = 20, 10, .5
  # sgdm = optim.SGD(model.parameters(), lr=sgdm_lr, momentum=0.9, weight_decay=1e-4)
  sgdm = optim.SGD([
    {'params': model.backbone.parameters(), 'lr': sgdm_lr * .1},
    {'params': model.scale.parameters(), 'lr': sgdm_lr},
    {'params': model.se_gpm.parameters(), 'lr': sgdm_lr},
    {'params': model.fc2cls.parameters(), 'lr': sgdm_lr}
  ], momentum=0.9, weight_decay=1e-4)
  schd = optim.lr_scheduler.CosineAnnealingLR(sgdm, T_max=schd_t + schd_d)

  # log
  smry_wrt = SummaryWriter(comment=f'-{model_id}-{expt.expt_id}')
  best_acc, best_epoch = .0, 0
  for e in range(epoch_cnt):
    # lr scheduling
    if e % schd_t == 0:
      for nlr in range(len(sgdm.param_groups)):
        sgdm.param_groups[nlr]['initial_lr'] = sgdm_lrs[nlr] * (schd_r ** (e // schd_t))
      # sgdm.param_groups[0]['initial_lr'] = sgdm_lr * (schd_r ** (e // schd_t))
      schd = optim.lr_scheduler.CosineAnnealingLR(sgdm, T_max=schd_t + schd_d)

    smry_wrt.add_scalars('Training/LR/epoch', {'backbone_lr': sgdm.param_groups[0]['lr'], 'module_lr':sgdm.param_groups[1]['lr']}, e)
    # training
    Loop.learn(expt, model, sgdm, schd, F.cross_entropy, 0, device_id, e, smry_wrt)
    _, accuracy = Loop.validate(expt, model, F.cross_entropy, 0, device_id, e, smry_wrt)

    # saving
    if accuracy > best_acc:
      best_acc = accuracy
      best_epoch = e
      with open(f'{model_id}-{expt.expt_id}.txt', 'w') as f:
        f.write(f'best_accuracy: {best_acc}% @ epoch: {best_epoch}')
      torch.save(model.state_dict(), f'saved/{model_id}-{expt.expt_id}.pth')

  smry_wrt.close()
