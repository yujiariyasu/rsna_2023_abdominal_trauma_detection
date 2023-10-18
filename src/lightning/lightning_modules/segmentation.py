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
        if self.cfg.mixup and (torch.rand(1)[0] < 0.5) and (self.cfg.warmup_epochs < self.current_epoch):
            mix_images, target_a, target_b, lam = mixup(images, targets, alpha=0.5)
            logits = self.forward(mix_images)
            loss = self.cfg.criterion(logits, target_a.float()) * lam + (1 - lam) * self.cfg.criterion(logits, target_b.float())
        else:
            logits = self.forward(images)
            loss = self.cfg.criterion(logits, targets.float())
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        images, targets = batch
        # np.save('tmp_mask', targets.detach().cpu().numpy())
        logits = self.forward(images)
        loss = self.cfg.criterion(logits, targets.float())
        output = OrderedDict({
            "targets": targets.detach().cpu(), "preds": logits.detach().cpu(), "loss": loss.detach()
        })
        return output

    def validation_epoch_end(self, outputs):
        d = dict()
        d["epoch"] = int(self.current_epoch)
        v_loss = torch.stack([o["loss"] for o in outputs]).mean().item()
        d["v_loss"] = v_loss

        targets = torch.cat([o["targets"] for o in outputs])#.cpu()#.numpy()
        preds = torch.cat([o["preds"] for o in outputs])#.cpu()#.numpy()
        if self.cfg.metric is None:
            score = -v_loss
        else:
            score = self.cfg.metric(targets, preds)
        d["val_metric"] = score
        self.log_dict(d, prog_bar=True)

        # st()

        # scores = []
        # if len(outputs)==2:
        #     d["val_metric"] = 0
        #     self.log_dict(d, prog_bar=True)
        #     return
        # N = 20
        # chunk = len(outputs)//N
        # for i in range(N):
        #     if i+1==N:
        #         targets = torch.cat([o["targets"] for o in outputs[i*chunk:]])#.cpu()#.numpy()
        #         preds = torch.cat([o["preds"] for o in outputs[i*chunk:]])#.cpu()#.numpy()
        #     else:
        #         targets = torch.cat([o["targets"] for o in outputs[i*chunk:(i+1)*chunk]])#.cpu()#.numpy()
        #         preds = torch.cat([o["preds"] for o in outputs[i*chunk:(i+1)*chunk]])#.cpu()#.numpy()
        #     if self.cfg.metric is None:
        #         score = -d['v_loss']
        #     # elif len(np.unique(targets)) == 1:
        #     #     score = 0
        #     else:
        #         score = self.cfg.metric(targets, preds)
        #     scores.append(score)
        # score = np.mean(scores)
        # d["val_metric"] = score
        # self.log_dict(d, prog_bar=True)

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
