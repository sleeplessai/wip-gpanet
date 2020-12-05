from res2pcs import Res2Pcs, Conv2dLeakyReluBnBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor


class Pcs2Cls(nn.Module):
  def __init__(
    self,
    model: nn.Module = None,
    class_cnt: int = 0,
    out_channels_per_layer: list = [64, 256, 512, 1024, 2048],
    out_mask: list = [0, 0, 1, 1, 1]
  ):
    assert len(out_mask) == len(out_mask)
    super(Pcs2Cls, self).__init__()

    self.res2u = model
    self.out_mask = np.array(out_mask, dtype=np.bool)
    self.res2_out = np.array(out_channels_per_layer)[self.out_mask].tolist()
    # self.res2_out = np.multiply(np.array(out_channels_per_layer), np.array(out_mask))
    print(self.res2_out)
    feature_size = np.sum(self.res2_out) // len(self.res2_out)


  def forward(self, x):
    features = np.array(self.res2u(x, feature_ext=True))[self.out_mask].tolist()
    out = []
    for o in out:
      print(o.size())


    return out


if __name__ == "__main__":
  res2pcs = Res2Pcs().cpu()
  pcs2cls = Pcs2Cls(model=res2pcs, class_cnt=102).cpu()
  # torch.save(Pcs2Cls.state_dict(), 'Pcs2Cls_init_null.pth')

  x = torch.rand((1, 3, 448, 448)).cpu()
  y = pcs2cls(x)
  # for o in out:
  #   print(o.size())
  # # print(cat_out.size())

