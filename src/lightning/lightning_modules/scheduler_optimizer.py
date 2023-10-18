from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, StepLR, CosineAnnealingLR
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch, math
from pdb import set_trace as st

def kaggle_note_fetch_scheduler(optimizer):
    scheduler = CosineAnnealingLR(optimizer,T_max=3565, eta_min=1e-6)
    return scheduler

import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict

class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # print(self.k)
        #assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """cosine annealing scheduler with warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer,
        T_max,
        eta_min,
        warmup_factor=1.0 / 3,
        warmup_epochs=2,
        warmup_method="linear",
        last_epoch=-1,
        resume_epoch=0
    ):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )

        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.resume_epoch = resume_epoch
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch+self.resume_epoch < self.warmup_epochs:
            return self.get_lr_warmup()
        else:
            return self.get_lr_cos_annealing()

    def get_lr_warmup(self):
        if self.warmup_method == "constant":
            warmup_factor = self.warmup_factor
        elif self.warmup_method == "linear":
            alpha = (self.last_epoch+self.resume_epoch) / self.warmup_epochs
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr * warmup_factor
            for base_lr in self.base_lrs
        ]

    def get_lr_cos_annealing(self):
        last_epoch = (self.last_epoch+self.resume_epoch) - self.warmup_epochs
        T_max = self.T_max - self.warmup_epochs
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * last_epoch / T_max)) / 2
                for base_lr in self.base_lrs]


def get_optimizer(cfg):
    if cfg.optimizer == 'adam':
        # optimizer = optim.Adam(cfg.model.parameters(), lr=cfg.lr)
        print('cfg.lr',cfg.lr)
        optimizer = optim.Adam(cfg.model.parameters(), lr=cfg.lr, weight_decay=1e-6)
    elif cfg.optimizer == 'radam':
        optimizer = optim.RAdam(cfg.model.parameters(), lr=cfg.lr)
    elif cfg.optimizer == 'lookahead_radam':
        optimizer = Lookahead(optim.RAdam(filter(lambda p: p.requires_grad, cfg.model.parameters()),lr=cfg.lr), alpha=0.5, k=5)
    elif cfg.optimizer == 'adamw':
        # optimizer = optim.AdamW(cfg.model.parameters(), lr=cfg.lr, weight_decay=cfg.lr/1.7)
        optimizer = optim.AdamW(cfg.model.parameters(), lr=cfg.lr)
    return optimizer

def get_nlp_optimizer(cfg):
    param_optimizer = list(cfg.model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': cfg.weight_decay},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if cfg.optimizer == 'adam':
        optim_class = optim.Adam
        # default: torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    elif cfg.optimizer == 'radam':
        optim_class = optim.RAdam
        # default: torch.optim.RAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        optim_class = AdamW

    if not hasattr(cfg, 'optimizer_betas'):
        cfg.optimizer_betas = (0.9, 0.999)
        # default: torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    optimizer = optim_class(optimizer_grouped_parameters,
                      lr=cfg.lr,
                      betas=cfg.optimizer_betas, # default (0.9, 0.999)
                      # betas=(0.9, 0.999), # default (0.9, 0.999)
                      weight_decay=cfg.weight_decay, # default 0
                      )

    return optimizer

def get_scheduler(cfg, optimizer):
    if cfg.scheduler == 'StepLR':
        scheduler = StepLR(optimizer, step_size=10000000, last_epoch=-1)
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, cooldown=1, verbose=True, min_lr=5e-7)
    elif cfg.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.t_max, eta_min=5e-7)
        # scheduler = CosineAnnealingLR(optimizer, T_max=cfg.t_max, eta_min=1e-7) # pao note
    elif cfg.scheduler == 'WarmupCosineAnnealingLR':
        scheduler = WarmupCosineAnnealingLR(optimizer, T_max=cfg.t_max, eta_min=cfg.eta_min, warmup_epochs=cfg.warmup_epochs, warmup_factor=0.1, resume_epoch=cfg.resume_epoch)
    elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.t_max, T_mult=1, eta_min=cfg.eta_min)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=min(cfg.epochs, 30), T_mult=1, eta_min=cfg.eta_min)
    elif cfg.scheduler == 'transformers_cosine':
        num_train_optimization_steps = int(cfg.len_train_loader * cfg.epochs)
        num_warmup_steps = int(num_train_optimization_steps * cfg.num_warmup_steps_rate)
        print('num_train_optimization_steps, num_warmup_steps:', num_train_optimization_steps, num_warmup_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_optimization_steps
        )
    elif cfg.scheduler == 'linear_schedule_with_warmup':
        num_train_optimization_steps = int(cfg.len_train_loader * cfg.epochs)
        num_warmup_steps = int(num_train_optimization_steps * cfg.num_warmup_steps_rate)
        print('num_train_optimization_steps, num_warmup_steps:', num_train_optimization_steps, num_warmup_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_optimization_steps
        )

    elif cfg.scheduler == 'timm_cosine':
        from timm.scheduler import CosineLRScheduler
        # https://blog.shikoan.com/timm-cosine-lr-scheduler/
        scheduler = CosineLRScheduler(optimizer, t_initial=cfg.epochs, lr_min=5e-7, t_mul=1.0,
                              warmup_t=1, warmup_lr_init=5e-5, warmup_prefix=True)
    elif cfg.scheduler == 'kaggle_note':
        scheduler = kaggle_note_fetch_scheduler(optimizer)
    return scheduler
