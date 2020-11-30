from res2pcs import Res2Pcs
# from pcs2cls import Pcs2Cls

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from fireblast.experiment.default import Experiment, default_cub200
from fireblast.experiment.loop import Loop
from fireblast.models.resnet import resnet50
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
  torch.cuda.empty_cache()
  model_id = 'resnet50'
  device_id = 'cuda:0'
  epoch_cnt = 160

  expt = Experiment()
  default_cub200(expt, data_loc='/home/i28/data/CUB_200_2011', loader=True)

  model = resnet50(pretrained=True)
  model.fc = nn.Linear(2048, expt.category_cnt, bias=False)

  sgdm = optim.SGD(model.parameters(), lr=.05/np.e, momentum=0.9, weight_decay=1e-4)
  sgdm_lr = sgdm.param_groups[0]['lr']
  schd_t, schd_d, schd_r = 20, 5, .5
  schd = optim.lr_scheduler.CosineAnnealingLR(sgdm, T_max=schd_t + schd_d)

  smry_wrt = SummaryWriter()

  best_acc, best_epoch = .0, 0
  for e in range(epoch_cnt):
    if e % schd_t == 0:
      sgdm.param_groups[0]['initial_lr'] = sgdm_lr * (schd_r ** (e // schd_t))
      schd = optim.lr_scheduler.CosineAnnealingLR(sgdm, T_max=schd_t + schd_d)

    # training
    smry_wrt.add_scalar('Training/LR/epoch', sgdm.param_groups[0]['lr'], e)
    Loop.learn(expt, model, sgdm, schd, F.cross_entropy, 0, device_id, e, smry_wrt)
    _, accuracy = Loop.validate(expt, model, F.cross_entropy, 0, device_id, e, smry_wrt)

    # saving
    if accuracy > best_acc:
      best_acc = accuracy
      best_epoch = e
      with open(f'{model_id}-{expt.expt_id}.txt', 'w') as f:
        f.write(f'best_accuracy: {best_acc}% @ epoch: {best_epoch}')
      torch.save(model.state_dict(), f'saved/{model_id}-{expt.expt_id}.pth')
