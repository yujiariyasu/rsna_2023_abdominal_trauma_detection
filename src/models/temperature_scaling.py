"""
Temperature Scaling
https://github.com/gpleiss/temperature_scaling
Modified
"""
import matplotlib.pyplot as plt
import torch
from sklearn.calibration import calibration_curve
from torch import nn, optim
from tqdm import tqdm


class TemperatureScaler(nn.Module):
    def __init__(self, model, cnn=False):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.cnn = cnn

    def forward(self, x):
        return self.__temperature_scale(self.model(x), inside=False)

    def __temperature_scale(self, logits, inside=True):
        if not inside and self.cnn:
            logits = logits[0]
        if not self.cnn:
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)

        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits.cuda() / temperature.cuda()

    def set_temperature(
        self, valid_loader,
    ):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """

        # First: collect all the logits and labels for the validation set
        logits = []
        labels = []
        with torch.no_grad():
            for x, y in tqdm(valid_loader, desc="calibration"):
                if self.cnn:
                    output, _ = self.model(x.half().cuda())
                else:
                    output = self.model(x.half().cuda())
                logits.append(output)
                labels.append(y)
            logits = torch.cat(logits)
            labels = torch.cat(labels).to(logits.device)

        if logits.size(1) == 2:
            nll_criterion = nn.CrossEntropyLoss()
            probas = logits.softmax(1)[:, 1]
        elif logits.size(1) == 1:
            nll_criterion = nn.BCEWithLogitsLoss()
            labels = labels.float().view(-1, 1)
            probas = logits.sigmoid()
        else:
            raise ValueError()

        # Calculate NLL and ECE before temperature scaling
        fop1, mpv1 = calibration_curve(
            labels.view(-1).cpu().numpy(), probas.cpu().numpy(), n_bins=10
        )
        print(fop1, mpv1)

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.__temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        logits2 = self.__temperature_scale(logits)
        if logits.size(1) == 2:
            probas2 = logits2.softmax(1)[:, 1]
        elif logits.size(1) == 1:
            probas2 = logits2.sigmoid()
        fop2, mpv2 = calibration_curve(
            labels.view(-1).cpu().numpy(), probas2.detach().cpu().numpy(), n_bins=10
        )
        print(fop2, mpv2)
        print("Optimal temperature: %.3f" % self.temperature.item())

        return self


class TemperatureScalerMetaMIL(nn.Module):
    def __init__(self, model, cnn=False):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.cnn = cnn

    def forward(self, x, meta):
        return self.__temperature_scale(self.model(x, meta), inside=False)

    def __temperature_scale(self, logits, inside=True):
        if not inside and self.cnn:
            logits = logits[0]
        if not self.cnn:
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits.cuda() / temperature.cuda()

    def set_temperature(
        self, valid_loader,
    ):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """

        # First: collect all the logits and labels for the validation set
        logits = []
        labels = []
        with torch.no_grad():
            for (x, x_meta), y in tqdm(valid_loader, desc="calibration"):
                if self.cnn:
                    output, _ = self.model(x.half().cuda(), x_meta.half().cuda())
                else:
                    output = self.model(x.half().cuda())
                logits.append(output)
                labels.append(y)
            logits = torch.cat(logits)
            labels = torch.cat(labels).to(logits.device)

        if logits.size(1) == 2:
            nll_criterion = nn.CrossEntropyLoss()
            probas = logits.softmax(1)[:, 1]
        elif logits.size(1) == 1:
            nll_criterion = nn.BCEWithLogitsLoss()
            labels = labels.float().view(-1, 1)
            probas = logits.sigmoid()
        else:
            raise ValueError()

        # Calculate NLL and ECE before temperature scaling
        fop1, mpv1 = calibration_curve(
            labels.view(-1).cpu().numpy(), probas.cpu().numpy(), n_bins=10
        )
        print(fop1, mpv1)

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.__temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        logits2 = self.__temperature_scale(logits)
        if logits.size(1) == 2:
            probas2 = logits2.softmax(1)[:, 1]
        elif logits.size(1) == 1:
            probas2 = logits2.sigmoid()
        fop2, mpv2 = calibration_curve(
            labels.view(-1).cpu().numpy(), probas2.detach().cpu().numpy(), n_bins=10
        )
        print(fop2, mpv2)
        print("Optimal temperature: %.3f" % self.temperature.item())

        return self
