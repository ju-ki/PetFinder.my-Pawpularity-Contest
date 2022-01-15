import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


def get_scheduler(optimizer, Config, num_train_steps=None):
    """
    if use get_linear_schedule_with_warmup or get_cosine_schedule_with_warmup, set num_train_steps.
    ex:
     num_train_steps = int(len(train_folds) / Config.batch_size * Config.epochs)

    if cosine or linear:
        num_warmup_steps (int) – The number of steps for the warmup phase.
        num_training_steps (int) – The total number of training steps.

        cosine only:

        num_cycles (float, optional, defaults to 0.5) – The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0 following a half-cosine).

    if ReduceLROnPlateau:
        mode (str) – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing
        factor (float) – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience (int) – Number of epochs with no improvement after which learning rate will be reduced.

    if CosineAnnealingLR:
        T_max (int) – Maximum number of iterations
        eta_min (float) – Minimum learning rate. Default: 0.

    if CosineAnnealingWarmRestarts:
        T_0 (int) – Number of iterations for the first restart
        T_mult (int, optional) – A factor increases T_{i}T after a restart. Default: 1.
        eta_min (float, optional) – Minimum learning rate. Default: 0.



    """
    if Config.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=Config.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif Config.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=Config.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=Config.num_cycles
        )
    elif Config.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=Config.factor, patience=Config.patience, verbose=True, eps=Config.eps)
    elif Config.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer, T_max=Config.T_max, eta_min=Config.min_lr, last_epoch=-1)
    elif Config.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=Config.T_0, T_mult=1, eta_min=Config.min_lr, last_epoch=-1)
    else:
        raise NotImplementedError
    return scheduler


def get_optimizer(model: nn.Module, Config: dict):
    if Config.optimizer_name == "AdamW":
        optimizer = AdamW(model.parameters(), lr=Config.lr,
                          weight_decay=Config.weight_decay)
    elif Config.optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=Config.lr,
                         weight_decay=Config.weight_decay)
    elif Config.optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=Config.lr,
                        weight_decay=Config.weight_decay)
    else:
        raise Exception('Unknown optimizer: {}'.format(Config.optimizer_name))
    return optimizer
