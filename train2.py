import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from fireblast.experiment.default import Experiment, default_dataloader
from fireblast.experiment.default import default_cub200, default_cars196, default_aircraft
from fireblast.experiment.loop import Loop
import fireblast.models as fbm
from torch.utils.tensorboard import SummaryWriter
from gpa2cls import gpa2cls_v1


gpu = torch.device('cuda')
expt2 = Experiment()
default_cub200(expt2, 'datasets/CUB_200_2011', True)
# default_cars196(expt2, 'datasets/cars196', True)
# default_aircraft(expt2, 'datasets/fgvc-aircraft-2013b', True, True)

model = gpa2cls_v1(cfg_file='gpa2cls-v1.yaml').to(gpu)
model_id = 'gpa2clsv1-resnet50'
max_epochs = 80
sgdm = optim.SGD(
  model.parameters(), lr=2e-3, momentum=0.9, weight_decay=1e-4, nesterov=True)
sche = optim.lr_scheduler.OneCycleLR(
  sgdm, max_lr=2e-3, epochs=max_epochs, steps_per_epoch=len(expt2.trainset_loader))

smry_wrt = SummaryWriter()
best_acc, best_epoch = 0, 0

for epoch in range(max_epochs):
  smry_wrt.add_scalar('Training/LR/epoch', sche.get_last_lr()[0], epoch)
  Loop.learn(expt2, model, sgdm, sche, F.cross_entropy, 2, gpu, epoch, smry_wrt)
  _, accuracy = Loop.validate(expt2, model, F.cross_entropy, 2, gpu, epoch, smry_wrt)
  accuracy = max(accuracy)
  if accuracy > best_acc:
    best_acc = accuracy
    best_epoch = epoch
    torch.save(model.state_dict(), f'{model_id}-{expt2.expt_id}.pth')
    with open('{model_id}-{expt2.expt_id}.txt', 'w') as f:
      f.write(f'best_accuracy: {best_acc:.6f}% @ epoch: {best_epoch:.6f}')

smry_wrt.close()
