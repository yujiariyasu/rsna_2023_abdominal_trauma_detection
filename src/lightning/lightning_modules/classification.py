from collections import OrderedDict
import torch.optim as optim

import pytorch_lightning as pl
import torch
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from pdb import set_trace as st
from .scheduler_optimizer import get_optimizer, get_scheduler

def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam

# def mixup(input, truth, clip=[0, 1]):
#     indices = torch.randperm(input.size(0))
#     shuffled_input = input[indices]
#     shuffled_labels = truth[indices]

#     lam = np.random.uniform(clip[0], clip[1])
#     input = input * lam + shuffled_input * (1 - lam)
#     return input, truth, shuffled_labels, lam


class MyLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super(MyLightningModule, self).__init__()
        self.model = cfg.model
        # if cfg.pretrained_path is not None:
        #     self.model.load_state_dict(torch.load(cfg.pretrained_path)['state_dict'])
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        images, targets = batch
        # print(images.size())
        if self.cfg.mixup and (torch.rand(1)[0] < 0.5) and (self.cfg.warmup_epochs < self.current_epoch) and (images.size(0) > 1):
            mix_images, target_a, target_b, lam = mixup(images, targets, alpha=0.5)
            if self.cfg.arcface:
                logits = self.model(mix_images, targets)
            else:
                logits = self.forward(mix_images)
                if self.cfg.distill:
                    with torch.no_grad():
                        for model_n, (model, weight) in enumerate(zip(self.cfg.teacher_models, [0.2, 0.4, 0.4])):
                            if model_n == 0:
                                teacher_preds = model(mix_images)*weight
                            else:
                                teacher_preds += model(mix_images)*weight
            if self.cfg.distill:
                if self.cfg.distill_cancer_only:
                    loss = self.cfg.criterion((logits[:, [0]]/self.cfg.distill_temperature).sigmoid(), (teacher_preds[:, [0]]/self.cfg.distill_temperature).sigmoid())
                else:
                    loss = self.cfg.criterion((logits/self.cfg.distill_temperature).sigmoid(), (teacher_preds/self.cfg.distill_temperature).sigmoid())
                if self.cfg.use_origin_label:
                    if self.cfg.criterion_for_origin_ratio == 0.5:
                        loss2 = self.cfg.criterion_for_origin(logits[:, 1:], targets[:, 1:])
                    else:
                        loss2 = self.cfg.criterion_for_origin(logits, targets)
                    loss = loss*self.cfg.criterion_for_origin_ratio + loss2*(1-self.cfg.criterion_for_origin_ratio)
            else:
                loss = self.cfg.criterion(logits, target_a) * lam + (1 - lam) * self.cfg.criterion(logits, target_b)
        else:
            if self.cfg.arcface:
                logits = self.model(images, targets)
            else:
                logits = self.forward(images)
                if self.cfg.distill:
                    with torch.no_grad():
                        for model_n, (model, weight) in enumerate(zip(self.cfg.teacher_models, [0.2, 0.4, 0.4])):
                            if model_n == 0:
                                teacher_preds = model(images)*weight
                            else:
                                teacher_preds += model(images)*weight
            if self.cfg.distill:
                if self.cfg.distill_cancer_only:
                    loss = self.cfg.criterion((logits[:, [0]]/self.cfg.distill_temperature).sigmoid(), (teacher_preds[:, [0]]/self.cfg.distill_temperature).sigmoid())
                else:
                    loss = self.cfg.criterion((logits/self.cfg.distill_temperature).sigmoid(), (teacher_preds/self.cfg.distill_temperature).sigmoid())
                if self.cfg.use_origin_label:
                    if self.cfg.criterion_for_origin_ratio == 0.5:
                        loss2 = self.cfg.criterion_for_origin(logits[:, 1:], targets[:, 1:])
                    else:
                        loss2 = self.cfg.criterion_for_origin(logits, targets)
                    loss = loss*self.cfg.criterion_for_origin_ratio + loss2*(1-self.cfg.criterion_for_origin_ratio)
            else:
                loss = self.cfg.criterion(logits, targets)
        # print(logits)
        # st()
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        images, targets = batch
        # if self.cfg.arcface:
        #     logits = self.model(images)
        # else:
        logits = self.forward(images)
        # torch.Size([32, 1, 512, 512])

        loss = self.cfg.criterion(logits, targets)
        # if isinstance(logits, tuple):
        #     # feature = logits[1]
        #     logits = logits[0]
        preds = logits
        # preds = logits.sigmoid()
        output = OrderedDict({
            "targets": targets.detach(), "preds": preds.detach(), "loss": loss.detach()
        })
        return output

    def validation_epoch_end(self, outputs):
        d = dict()
        d["epoch"] = int(self.current_epoch)
        d["v_loss"] = torch.stack([o["loss"] for o in outputs]).mean().item()

        targets = torch.cat([o["targets"] for o in outputs]).cpu()#.numpy()
        preds = torch.cat([o["preds"] for o in outputs]).cpu()#.numpy()
        if self.cfg.metric is None:
            score = -d['v_loss']
        elif len(np.unique(targets)) == 1:
            score = 0
        else:
            # score = self.cfg.metric(targets, preds)
            score = self.cfg.metric(targets, preds)
        d["val_metric"] = score
        np.save(f'{self.cfg.output_path}/val_preds/fold{self.cfg.fold}/epoch{self.current_epoch}.npy', preds)
        self.log_dict(d, prog_bar=True)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": get_scheduler(self.cfg, optimizer),
                "monitor": 'val_metric',
                "frequency": 1
            }
        }

    # def configure_optimizers(self):
    #     optimizer = get_optimizer(self.cfg)

    #     # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
    #     self.lr_scheduler = CosineWarmupScheduler(
    #         optimizer, warmup=2, max_iters=20
    #     )
    #     return optimizer

    # def optimizer_step(self, *args, **kwargs):
    #     super().optimizer_step(*args, **kwargs)
    #     self.lr_scheduler.step()  # Step per iteration


    # learning rate warm-up
    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
    #     # warm up lr
    #     if self.trainer.global_step < 500:
    #         lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = lr_scale * self.hparams.learning_rate

    #     # update params
    #     optimizer.step(closure=closure)

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
