"""Warmup + cosine LR schedule, ported from EAD_navsim
(navsim/agents/ead/modules/scheduler.py).

Behaviour is EAD's verbatim: a linear warmup over ``warmup_epochs`` reaching the
full ``lr``, then a cosine decay to ``min_lr`` over the remaining
``epochs - warmup_epochs``.

One adaptation matters here. EAD's ``get_lr`` ends with

    return [lr for _ in self.optimizer.param_groups]

i.e. every parameter group is driven to the SAME learning rate, which would wipe
out this repo's per-group multipliers (``lr_mult_backbone`` for the BEV
front-end, ``lr_mult_img_backbone`` for the pretrained image backbone -- the 0.1x
that keeps fine-tuning from destroying ViT-L / V2-99). EAD's own escape hatch is
the ``lr_scale`` key, which it only honours when group 0 carries it, so
``DriveSuprimAgent.get_optimizers`` tags EVERY group with ``lr_scale`` and the
multipliers survive the schedule.
"""

import math

from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosLR(_LRScheduler):
    def __init__(self, optimizer, min_lr, lr, warmup_epochs, epochs,
                 last_epoch=-1, verbose=False) -> None:
        self.min_lr = min_lr
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        # PyTorch changed the base scheduler signature and removed `verbose`
        # from newer `LRScheduler` initializers.
        try:
            super(WarmupCosLR, self).__init__(optimizer, last_epoch=last_epoch,
                                              verbose=verbose)
        except TypeError:
            super(WarmupCosLR, self).__init__(optimizer, last_epoch=last_epoch)
            self.verbose = verbose

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_init_lr(self):
        return self.lr / self.warmup_epochs

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1 + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs)
                    / (self.epochs - self.warmup_epochs)
                )
            )
        if "lr_scale" in self.optimizer.param_groups[0]:
            return [lr * group["lr_scale"] for group in self.optimizer.param_groups]
        return [lr for _ in self.optimizer.param_groups]
