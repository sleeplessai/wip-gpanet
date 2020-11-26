from res2pcs import Res2Pcs
# from pcs2cls import Pcs2Cls

import torch
import torch.optim as optim
import torch.nn.functional as F

from fireblast.experiment.default import *
from fireblast.experiment.loop import Loop


if __name__ == "__main__":
  torch.manual_seed(11295)
  torch.cuda.empty_cache()

  model_id = 'res2pcs-50-4x16p-minisc'
  task_id = 'cars196'
  device_id = 'cuda:0'
  epoch_cnt = 220

  res2pcs = Res2Pcs(
    block_conf_per_layer=[
      (64, 256, 16),
      (256, 512, 16),
      (512, 1024, 16),
      (1024, 2048, 16)
      ],
    class_cnt=196,
    mini_identity_shortcut=True
  )
  with open(f'{model_id}-{task_id}.txt', 'w') as f:
    f.write(str(res2pcs))
  torch.save(res2pcs.state_dict(), f'{model_id}.pth') # save to observe parameters

  fbxo = FireblastExperiment()
  default_cars196(fbxo)
  train_loader = torch.utils.data.DataLoader(
    dataset=fbxo.Trainset,
    batch_size=12,
    shuffle=True,
    num_workers=6
  )
  valid_loader = torch.utils.data.DataLoader(
    dataset=fbxo.Testset,
    batch_size=6,
    shuffle=False,
    num_workers=6
  )

  optimizer = optim.SGD(res2pcs.parameters(), lr=2e-4, momentum=0.9, weight_decay=1e-4)
  lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=2e-4, max_lr=1.6e-2, step_size_up=10, step_size_down=12)

  best_epoch, best_accuracy = None, 0.
  swrt = torch.utils.tensorboard.SummaryWriter(comment=f'-{model_id}-{task_id}')

  for epoch in range(epoch_cnt):
    swrt.add_scalar('Training/Learning Rate/epoch', optimizer.param_groups[0]['lr'], epoch + 1)
    Loop.learn(train_loader, res2pcs, optimizer, lr_scheduler, F.cross_entropy, device_id, epoch + 1, swrt)
    _, accuracy = Loop.validate(valid_loader, res2pcs, F.cross_entropy, device_id, epoch + 1, swrt)
    if accuracy > best_accuracy:
      torch.save(res2pcs.state_dict(), f'saved/{model_id}-{task_id}-best.pth')
      best_accuracy = accuracy
      best_epoch = epoch + 1
  swrt.close()
  print(f'best_accuracy@epoch: {best_accuracy:.4f}% @ {best_epoch}')

