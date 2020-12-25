import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class Conv2dNormLeakyReluBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
    super(Conv2dNormLeakyReluBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    self.bn = nn.BatchNorm2d(out_channels)

  def forward(self, x, activate=True):
    x = self.conv(x)
    x = self.bn(x)
    if activate:
      x = F.leaky_relu(x, negative_slope=0.1)
    return x


class Res2PcsBlock(nn.Module):
  def __init__(self, in_channels, out_channels, pieces, mini_shortcut=False):
    super(Res2PcsBlock, self).__init__()
    self.pieces = pieces
    self.conv1 = Conv2dNormLeakyReluBlock(in_channels, in_channels, kernel_size=1)
    piece_size = in_channels // pieces
    assert piece_size * pieces == in_channels
    self.conv2 = nn.ModuleList()
    for _ in range(self.pieces):
      self.conv2.append(Conv2dNormLeakyReluBlock(piece_size * 2, piece_size, kernel_size=3, padding=1))
    self.conv3 = Conv2dNormLeakyReluBlock(in_channels, out_channels, kernel_size=1)
    self.mini_shortcut = mini_shortcut
    self.init_weights()

  def init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    identity = x
    x = self.conv1(x)
    x_pieces = list(torch.chunk(x, self.pieces, dim=1))
    for i in range(len(x_pieces)):
      mini_identity = x_pieces[i]
      if i == 0:
        x_pieces[i] = torch.cat((x_pieces[i], x_pieces[-1]), dim=1)
      else:
        x_pieces[i] = torch.cat((x_pieces[i], x_pieces[i - 1]), dim=1)
      if self.mini_shortcut:
        x_pieces[i] = self.conv2[i](x_pieces[i], activate=False)
        x_pieces[i] += mini_identity
        x_pieces[i] = F.leaky_relu(x_pieces, negative_slope=0.1)
      else:
        x_pieces[i] = self.conv2[i](x_pieces[i])

    x = torch.cat(x_pieces, dim=1)
    x = self.conv3(x, activate=False)
    x += identity
    x = F.leaky_relu(x, negative_slope=0.1)
    return x


class _Transparent(nn.Module):
  def __init__(self): super(_Transparent, self).__init__()
  def forward(self, x): return x

class Res2Pcs(nn.Module):
  def __init__(
    self,
    resnet_conv7x7: bool = False,
    block_cnt_per_layer: list = [3, 4, 6, 3],
    block_conf_per_layer: list = [(64, 256, 8), (256, 512, 8), (512, 1024, 8), (1024, 2048, 8)],
    mini_identity_shortcut: bool = False,
    class_cnt: int = 0
  ):
    super(Res2Pcs, self).__init__()

    if resnet_conv7x7:
      self.conv1 = Conv2dNormLeakyReluBlock(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
      self.conv1 = nn.Sequential(
        Conv2dNormLeakyReluBlock(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        Conv2dNormLeakyReluBlock(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
      )

    mini_sc = mini_identity_shortcut
    self.layers = nn.ModuleList()
    for block_cnt, block_conf in zip(block_cnt_per_layer, block_conf_per_layer):
      stage = nn.Sequential()
      for i in range(block_cnt):
        block_id = "Res2PcsBlock_" + str(i)
        if i < block_cnt - 1:
          stage.add_module(block_id, Res2PcsBlock(block_conf[0], block_conf[0], block_conf[2], mini_shortcut=mini_sc))
        else:
          stage.add_module(block_id, Res2PcsBlock(block_conf[0], block_conf[1], block_conf[2], mini_shortcut=mini_sc))
      self.layers.append(stage)

    if class_cnt:
      self.fc = nn.Linear(block_conf_per_layer[-1][1], class_cnt)
    else:
      self.fc = _Transparent()

  def forward(self, x: Tensor, feature_ext=False):
    out = []
    x = self.conv1(x)
    x = torch.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    out.append(x)
    # print(x.size())
    for stage in self.layers:
      x = stage(x)
      x = torch.max_pool2d(x, kernel_size=3, stride=2, padding=1)
      out.append(x)
      # print(x.size())
    if feature_ext:
      return out
    x = F.adaptive_avg_pool2d(x, 1)
    x = torch.flatten(x, 1)
    # print(x.size())
    x = self.fc(x)
    # print(x.size())
    return x
