import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from fireblast.experiment.experiment import Experiment, set_cuda_visible_devices
from fireblast.experiment.default import default_cub200, default_cars196, default_aircraft
from fireblast.experiment.loop import Loop

from modeling import gpa2cls_v1a, gpa2cls_v1b


if __name__ == "__main__":
    set_cuda_visible_devices(0)
    expt2 = Experiment()

    default_cub200(expt2, 'datasets/CUB_200_2011', True, batch_size=(12, 6))
    # default_cars196(expt2, 'datasets/cars196', True)
    # default_aircraft(expt2, 'datasets/fgvc-aircraft-2013b', True, True)

    model2 = gpa2cls_v1a('configs/gpa2cls-v1-r50-2560d2x2.yaml', num_classes=expt2.category_cnt).cuda()
    logging.warning(model2.model_id)

    max_epochs = 80
    sgdm = optim.SGD(
        model2.parameters(), lr=2e-3, momentum=0.9, weight_decay=1e-4, nesterov=True)
    sche = optim.lr_scheduler.OneCycleLR(
        sgdm, max_lr=2e-3, epochs=max_epochs, steps_per_epoch=len(expt2.trainset_loader))

    smry_wrt = SummaryWriter(comment=f'-{model2.model_id}-{expt2.expt_id}')
    best_acc, best_epoch = 0, 0

    for epoch in range(max_epochs):
        smry_wrt.add_scalar('Training/LR/epoch', sche.get_last_lr()[0], epoch)
        Loop.learn(expt2, model2, sgdm, sche, F.cross_entropy, 2, epoch=epoch, summary_writer=smry_wrt)
        _, accuracy = Loop.validate(
            expt2, model2, F.cross_entropy, 2, epoch=epoch, summary_writer=smry_wrt, stdout_freq=100)
        accuracy = max(accuracy)
        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = epoch
            torch.save(model2.state_dict(), f'saved/{model2.model_id}-{expt2.expt_id}.pth')
            with open(f'best/{model2.model_id}-{expt2.expt_id}.txt', 'w') as f:
                f.write(f'best_accuracy: {best_acc:.6f}% @ epoch: {best_epoch}')

    smry_wrt.close()
