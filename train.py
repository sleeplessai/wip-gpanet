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

  model_id = 'res2pcs-50'
  device_id = 'cuda:0'
  epoch_cnt = 220

  res2pcs = Res2Pcs(
    block_conf_per_layer=[
      (64, 256, 16),
      (256, 512, 16),
      (512, 1024, 16),
      (1024, 2048, 16)
      ],
    class_cnt=102,
    mini_identity_shortcut=True
  )
  with open(f'{model_id}.txt', 'w') as f:
    f.write(str(res2pcs))
  torch.save(res2pcs.state_dict(), f'{model_id}.pth') # save to observe parameters
