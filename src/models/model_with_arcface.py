import torch
import torch.nn as nn
import timm
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
import sys
from pdb import set_trace as st
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ArcMarginProductOutCosine(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine

class ArcMarginProductSubcenterOutCosine(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine

class AddMarginProduct(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        if label is not None:
            one_hot = torch.zeros(cosine.size(), device=device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            output = cosine

        output *= self.s
        # print(output)

        return output

class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine.float() > self.th, phi, cosine.float() - self.mm)
        if label is not None:
            one_hot = torch.zeros(cosine.size(), device=device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            output = cosine
        output *= self.s

        return output

class ArcMarginProductLabelOnehot(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProductLabelOnehot, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine.float() > self.th, phi, cosine.float() - self.mm)
        if label is not None:
            one_hot = label.long()
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            output = cosine
        output *= self.s

        return output

class ArcMarginProductSubcenter(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, k=3):
        super(ArcMarginProductSubcenter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.k = k
        # 以下どちらも試す
        nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine_all = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        if label is not None:
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine.float() > self.th, phi, cosine.float() - self.mm)
            one_hot = torch.zeros(cosine.size(), device=device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            output = cosine
        output *= self.s

        return output

sigmoid = nn.Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class EnetBertArcface(nn.Module):
    def __init__(self, model_name, pretrained, out_dim):
        super(EnetBertArcface, self).__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained('input/bert-base-uncased')
        self.enet = timm.create_model(model_name, pretrained=pretrained)
#         output = net.fc.out_features

        self.feat = nn.Linear(self.enet.classifier.in_features+self.bert.config.hidden_size, 512)
        self.swish = Swish_module()
        self.dropout = nn.Dropout(0.5)
        self.metric_classify = ArcMarginProduct(512, out_dim)
#         self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.classifier = nn.Identity()

    def forward(self, x,input_ids, attention_mask, label=None):
        x = self.enet(x)
        text = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        x = torch.cat([x, text], 1)
        features = self.swish(self.feat(x))
        x = self.metric_classify(features, label)
        return F.normalize(features), x

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GaP(nn.Module):
    def __init__(self):
        super(GaP, self).__init__()

    def forward(self, x):
        return x.mean(axis=-1).mean(axis=-1)

class PudaeArcNet(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=1, s=30, m=0.5, k=3, use_gem=True, use_dropout=False, use_subcenter=True, out_cosine=True):
        super(PudaeArcNet, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
        self.use_dropout = use_dropout
        # assert('swin' in model_name)
        if 'swin' in model_name:
            in_features = self.model.head.in_features
            sys.path.append('/home/acc12347av/ml_pipeline/src')
            import timm055
            self.model = timm055.create_model(model_name, pretrained=pretrained, num_classes=1, arcface=True, skip_attn=True, img_size=(768,768))
            self.model.avgpool = nn.Identity()
            self.model.head = nn.Identity()
            self.pooling = nn.AdaptiveAvgPool1d(1)
        elif 'beit' in model_name:
            in_features = self.model.head.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.avgpool = nn.Identity()
            self.model.head = nn.Identity()
            self.model.fc_norm = nn.Identity()
            self.pooling = nn.Identity()
        elif 'convnext' in model_name:
            in_features = self.model.head.fc.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.head = nn.Identity()
            if use_gem:
                self.pooling = GeM()
            else:
                self.pooling = GaP()
        else:
            in_features = self.model.classifier.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.classifier = nn.Identity()
            self.model.global_pool = nn.Identity()
            if use_gem:
                self.pooling = GeM()
            else:
                self.pooling = GaP()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout2d(0.5, inplace=True)
        self.fc1 = nn.Linear(in_features, 512)
        self.bn2 = nn.BatchNorm1d(512)

        if out_cosine:
            if use_subcenter:
                self.metric_classify = ArcMarginProductSubcenterOutCosine(512,
                                           num_classes,
                                           k=k)
            else:
                self.metric_classify = ArcMarginProductOutCosine(512,
                                           num_classes)
        else:
            if use_subcenter:
                self.metric_classify = ArcMarginProductSubcenter(512,
                                           num_classes,
                                           s=s,
                                           m=m,
                                           k=k)
            else:
                self.metric_classify = ArcMarginProduct(512,
                                           num_classes,
                                           s=s,
                                           m=m,
                                           easy_margin=False)

    def forward(self, images, labels=None):
        features = self.model(images)
        features = self.pooling(features).flatten(1)
        features = self.bn1(features)
        features = self.dropout(features)
        features = self.fc1(features)
        features = self.bn2(features)
        features = F.normalize(features)
        output = self.metric_classify(features, labels)

        if self.training:
            return output
        else:
            return output, features

class WithArcface(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=1, s=30, m=0.5, k=3, use_gem=True, dropout_ratio=None, use_subcenter=True, out_cosine=True, label_onehot=False, use_layer_norm=False):
        super(WithArcface, self).__init__()
        self.use_layer_norm = use_layer_norm
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
        self.dropout_ratio = dropout_ratio
        self.label_onehot = label_onehot
        self.out_cosine = out_cosine
        if 'swin' in model_name:
            in_features = self.model.head.in_features
            sys.path.append('/home/acc12347av/ml_pipeline/src')
            import timm055
            self.model = timm055.create_model(model_name, pretrained=pretrained, num_classes=1, arcface=True, skip_attn=True, img_size=(384*2,384*2))
            self.model.avgpool = nn.Identity()
            self.model.head = nn.Identity()
            self.pooling = nn.AdaptiveAvgPool1d(1)
        elif 'vit' in model_name:
            in_features = self.model.head.in_features
            sys.path.append('/home/acc12347av/ml_pipeline/src')
            import timm055
            self.model = timm055.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.avgpool = nn.Identity()
            self.model.head = nn.Identity()
            self.pooling = nn.Identity()
        elif 'beit' in model_name:
            in_features = self.model.head.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.avgpool = nn.Identity()
            self.model.head = nn.Identity()
            self.model.fc_norm = nn.Identity()
            self.pooling = nn.Identity()
        elif 'convnext' in model_name:
            in_features = self.model.head.fc.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.head = nn.Identity()
            if use_gem:
                self.pooling = GeM()
            else:
                self.pooling = GaP()
        else:
            in_features = self.model.classifier.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.classifier = nn.Identity()
            self.model.global_pool = nn.Identity()
            if use_gem:
                self.pooling = GeM()
            else:
                self.pooling = GaP()
        if self.dropout_ratio is not None:
            self.dropout = nn.Dropout(self.dropout_ratio)
        self.fc = nn.Linear(in_features, 512)
        if self.use_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(512)
        # print('infeatures:', in_features)
        if out_cosine:
            if use_subcenter:
                self.metric_classify = ArcMarginProductSubcenterOutCosine(512,
                                           num_classes,
                                           k=k)
            else:
                self.metric_classify = ArcMarginProductOutCosine(512,
                                           num_classes)
        else:
            if use_subcenter:
                self.metric_classify = ArcMarginProductSubcenter(512,
                                           num_classes,
                                           s=s,
                                           m=m,
                                           k=k)
            else:
                if self.label_onehot:
                    self.metric_classify = ArcMarginProductLabelOnehot(512,
                                               num_classes,
                                               s=s,
                                               m=m,
                                               easy_margin=False)
                else:
                    self.metric_classify = ArcMarginProduct(512,
                                               num_classes,
                                               s=s,
                                               m=m,
                                               easy_margin=False)

    def forward(self, images, labels=None):
        features = self.model(images)
        # print('before',features.size())
        features = self.pooling(features).flatten(1)
        # print('after',features.size())
        if self.dropout_ratio is not None:
            features = self.dropout(features)
        features = self.fc(features)
        if self.use_layer_norm:
            features = self.layer_norm(features)
        if self.out_cosine:
            output = self.metric_classify(features)
        else:
            output = self.metric_classify(features, labels)

        if self.training:
            return output
        else:
            return output, features

from torchvision import transforms
class Guie(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=1, s=30, m=0.5, k=3, use_gem=True, dropout_ratio=None, use_subcenter=True, out_cosine=True, label_onehot=False, use_layer_norm=True):
        super(Guie, self).__init__()
        self.use_layer_norm = use_layer_norm
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
        self.dropout_ratio = dropout_ratio
        self.label_onehot = label_onehot
        self.out_cosine = out_cosine
        if 'swin' in model_name:
            in_features = self.model.head.in_features
            sys.path.append('/home/acc12347av/ml_pipeline/src')
            import timm055
            self.model = timm055.create_model(model_name, pretrained=pretrained, num_classes=1, arcface=True, skip_attn=True, img_size=(384,384))
            # self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1, arcface=True, skip_attn=True, img_size=(384*2,384*2))
            self.model.avgpool = nn.Identity()
            self.model.head = nn.Identity()
            self.pooling = nn.AdaptiveAvgPool1d(1)
        elif 'vit' in model_name:
            in_features = self.model.head.in_features
            sys.path.append('/home/acc12347av/ml_pipeline/src')
            import timm055
            self.model = timm055.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.avgpool = nn.Identity()
            self.model.head = nn.Identity()
            self.pooling = nn.Identity()
        elif 'beit' in model_name:
            in_features = self.model.head.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.avgpool = nn.Identity()
            self.model.head = nn.Identity()
            self.model.fc_norm = nn.Identity()
            self.pooling = nn.Identity()
        elif 'convnext' in model_name:
            in_features = self.model.head.fc.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.head = nn.Identity()
            if use_gem:
                self.pooling = GeM()
            else:
                self.pooling = GaP()
        else:
            in_features = self.model.classifier.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.classifier = nn.Identity()
            self.model.global_pool = nn.Identity()
            if use_gem:
                self.pooling = GeM()
            else:
                self.pooling = GaP()
        if self.dropout_ratio is not None:
            self.dropout = nn.Dropout(self.dropout_ratio)
        self.fc = nn.Linear(in_features, 64)
        if self.use_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(64)

        if out_cosine:
            if use_subcenter:
                self.metric_classify = ArcMarginProductSubcenterOutCosine(64,
                                           num_classes,
                                           k=k)
            else:
                self.metric_classify = ArcMarginProductOutCosine(64,
                                           num_classes)
        else:
            if use_subcenter:
                self.metric_classify = ArcMarginProductSubcenter(64,
                                           num_classes,
                                           s=s,
                                           m=m,
                                           k=k)
            else:
                if self.label_onehot:
                    self.metric_classify = ArcMarginProductLabelOnehot(64,
                                               num_classes,
                                               s=s,
                                               m=m,
                                               easy_margin=False)
                else:
                    self.metric_classify = ArcMarginProduct(64,
                                               num_classes,
                                               s=s,
                                               m=m,
                                               easy_margin=False)

    def forward(self, images, labels=None):
        images = transforms.functional.resize(images, size=[384, 384])
        images = images/255.0
        images = transforms.functional.normalize(images,
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

        features = self.model(images)
        # print('before',features.size())
        features = self.pooling(features).flatten(1)
        # print('after',features.size())
        if self.dropout_ratio is not None:
            features = self.dropout(features)
        features = self.fc(features)
        if self.use_layer_norm:
            features = self.layer_norm(features)

        if self.out_cosine:
            output = self.metric_classify(features)
        else:
            output = self.metric_classify(features, labels)

        return output

    def inf(self, images, labels=None):
        images = transforms.functional.resize(images, size=[384, 384])
        images = images/255.0
        images = transforms.functional.normalize(images,
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

        features = self.model(images)
        # print('before',features.size())
        features = self.pooling(features).flatten(1)
        # print('after',features.size())
        if self.dropout_ratio is not None:
            features = self.dropout(features)
        features = self.fc(features)
        if self.use_layer_norm:
            features = self.layer_norm(features)
        if not self.training:
            return features

        if self.out_cosine:
            output = self.metric_classify(features)
        else:
            output = self.metric_classify(features, labels)

        return output

class Guie2(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=1):
        super(Guie2, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=64)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, images, labels=None):
        images = transforms.functional.resize(images, size=[384, 384])
        images = images/255.0
        images = transforms.functional.normalize(images,
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

        features = self.model(images)
        return features

    def inf(self, images):
        images = transforms.functional.resize(images, size=[384, 384])
        images = images/255.0
        images = transforms.functional.normalize(images,
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

        features = self.model(images)
        return self.fc(features)


def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
class MultiAtrousModule(nn.Module):
    def __init__(self, in_chans, out_chans, dilations):
        super(MultiAtrousModule, self).__init__()

        self.d0 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[0],padding='same')
        self.d1 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[1],padding='same')
        self.d2 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[2],padding='same')
        self.conv1 = nn.Conv2d(512 * 3, out_chans, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self,x):

        x0 = self.d0(x)
        x1 = self.d1(x)
        x2 = self.d2(x)
        x = torch.cat([x0,x1,x2],dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        return x

class SpatialAttention2d(nn.Module):
    def __init__(self, in_c):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 1024, 1, 1)
        self.bn = nn.BatchNorm2d(1024)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20) # use default setting.

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score
        '''
        x = self.conv1(x)
        x = self.bn(x)

        feature_map_norm = F.normalize(x, p=2, dim=1)

        x = self.act1(x)
        x = self.conv2(x)
        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)

        x = att * feature_map_norm
        return x, att_score

class OrthogonalFusion(nn.Module):
    def __init__(self):
        super(OrthogonalFusion, self).__init__()

    def forward(self, fl, fg):

        bs, c, w, h = fl.shape

        fl_dot_fg = torch.bmm(fg[:,None,:],fl.reshape(bs,c,-1))
        fl_dot_fg = fl_dot_fg.reshape(bs,1,w,h)
        fg_norm = torch.norm(fg, dim=1)

        fl_proj = (fl_dot_fg / fg_norm[:,None,None,None]) * fg[:,:,None,None]
        fl_orth = fl - fl_proj

        f_fused = torch.cat([fl_orth,fg[:,:,None,None].repeat(1,1,w,h)],dim=1)
        return f_fused

class ArcMarginProduct_subcenter_dolg(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine

class WhalePrev1stModel(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=1, s=30, m=0.5, k=3, use_gem=True, use_dropout=False, use_subcenter=True, out_cosine=True, stride=None, pool='gem', embedding_size=512, arcface_s=45, arcface_m=0.3, dilations=[3,6,9]):
        super(WhalePrev1stModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
        self.use_dropout = use_dropout
        if 'swin' in model_name:
            in_features = self.model.head.in_features
            sys.path.append('/home/acc12347av/ml_pipeline/src')
            import timm055
            self.model = timm055.create_model(model_name, pretrained=pretrained, num_classes=1, arcface=True, skip_attn=True, img_size=(768,768))
            self.model.avgpool = nn.Identity()
            self.model.head = nn.Identity()
            self.pooling = nn.AdaptiveAvgPool1d(1)
        elif 'beit' in model_name:
            in_features = self.model.head.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.avgpool = nn.Identity()
            self.model.head = nn.Identity()
            self.model.fc_norm = nn.Identity()
            self.pooling = nn.Identity()
        elif 'convnext' in model_name:
            in_features = self.model.head.fc.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.head = nn.Identity()
            if use_gem:
                self.pooling = GeM()
            else:
                self.pooling = GaP()
        else:
            in_features = self.model.classifier.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.classifier = nn.Identity()
            self.model.global_pool = nn.Identity()
            if use_gem:
                self.pooling = GeM()
            else:
                self.pooling = GaP()

        local_planes = 512
        self.local_conv = nn.Conv2d(in_features, local_planes, 1)
        self.local_bn = nn.BatchNorm2d(local_planes)
        self.local_bn.bias.requires_grad_(False)  # no shift
        self.bottleneck_g = nn.BatchNorm1d(in_features)
        self.bottleneck_g.bias.requires_grad_(False)  # no shift
        # self.fc = nn.Linear(in_features, num_classes)
        # nn.init.normal_(self.fc.weight, std=0.001)
        # nn.init.constant_(self.fc.bias, 0)

        feature_dim_l_g = 1024
        fusion_out = 2 * feature_dim_l_g
        pool='gem'
        if pool == "gem":
            self.global_pool = GeM(p_trainable=True)
        elif pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fusion_pool = nn.AdaptiveAvgPool2d(1)

        self.neck = nn.Sequential(
                nn.Linear(fusion_out, embedding_size, bias=True),
                nn.BatchNorm1d(embedding_size),
                torch.nn.PReLU()
            )

        self.head_in_units = embedding_size
        self.head = ArcMarginProduct_subcenter_dolg(embedding_size, num_classes)
        # if loss == 'adaptive_arcface':
        #     self.loss_fn = ArcFaceLossAdaptiveMargin(dataset.margins, num_classes, arcface_s)
        # elif loss == 'arcface':
        #     self.loss_fn = ArcFaceLoss(arcface_s, arcface_m)
        # else:
        #     pass
        backbone_out_1=10
        backbone_out=20
        self.mam = MultiAtrousModule(backbone_out_1, feature_dim_l_g, dilations)
        self.conv_g = nn.Conv2d(backbone_out, feature_dim_l_g, kernel_size=1)
        self.bn_g = nn.BatchNorm2d(feature_dim_l_g, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act_g = nn.SiLU(inplace=True)
        self.attention2d = SpatialAttention2d(feature_dim_l_g)
        self.fusion = OrthogonalFusion()




    def forward(self, x, label=None):
        feat = self.model(x)
        # global feat
        global_feat = F.avg_pool2d(feat, feat.size()[2:])
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = F.dropout(global_feat, p=0.2)
        global_feat = self.bottleneck_g(global_feat)
        global_feat = l2_norm(global_feat)

        # local feat
        local_feat = torch.mean(feat, -1, keepdim=True)
        local_feat = self.local_bn(self.local_conv(local_feat))
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        local_feat = l2_norm(local_feat, axis=-1)

        local_feat = self.mam(local_feat)
        local_feat, att_score = self.attention2d(local_feat)

        global_feat = self.conv_g(global_feat)
        global_feat = self.bn_g(global_feat)
        global_feat = self.act_g(global_feat)

        global_feat = self.global_pool(global_feat)
        global_feat = global_feat[:,:,0,0]

        x_fused = self.fusion(local_feat, global_feat)
        x_fused = self.fusion_pool(x_fused)
        x_fused = x_fused[:,:,0,0]

        x_emb = self.neck(x_fused)

        # if self.headless:
        #     return {"target": batch['target'],'embeddings': x_emb}

        logits = self.head(x_emb)
        return logits

        # out = self.fc(global_feat) * 20
        # if self.training:
        #     return [global_feat, local_feat, out]
        # else:
        #     return out, global_feat

class WithArcfaceDivideFeatures(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=1, s=30, m=0.5, k=3, use_gem=True, use_dropout=False, use_subcenter=True, out_cosine=True):
        super(WithArcfaceDivideFeatures, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
        self.use_dropout = use_dropout
        self.out_cosine = out_cosine
        # assert('swin' in model_name)
        if 'swin' in model_name:
            in_features = self.model.head.in_features
            sys.path.append('/home/acc12347av/ml_pipeline/src')
            import timm055
            self.model = timm055.create_model(model_name, pretrained=pretrained, num_classes=1, arcface=True, skip_attn=True, img_size=(768,768))
            self.model.avgpool = nn.Identity()
            self.model.head = nn.Identity()
            self.pooling = nn.AdaptiveAvgPool1d(1)
        elif 'beit' in model_name:
            in_features = self.model.head.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.avgpool = nn.Identity()
            self.model.head = nn.Identity()
            self.model.fc_norm = nn.Identity()
            self.pooling = nn.Identity()
        elif 'convnext' in model_name:
            in_features = self.model.head.fc.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.head = nn.Identity()
            if use_gem:
                self.pooling = GeM()
            else:
                self.pooling = GaP()
        else:
            in_features = self.model.classifier.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.classifier = nn.Identity()
            self.model.global_pool = nn.Identity()
            if use_gem:
                self.pooling = GeM()
            else:
                self.pooling = GaP()
        if self.use_dropout:
            self.dropout = nn.Dropout(0.2)
        self.fc0 = nn.Linear(in_features, in_features//4)
        self.fc1 = nn.Linear(in_features, in_features//4)
        self.fc2 = nn.Linear(in_features, in_features//4)
        self.fc3 = nn.Linear(in_features, in_features//4)
        # print('infeatures:', in_features)
        if out_cosine:
            if use_subcenter:
                self.metric_classify = ArcMarginProductSubcenterOutCosine(512,
                                           num_classes,
                                           k=k)
            else:
                self.metric_classify = ArcMarginProductOutCosine(512,
                                           num_classes)
        else:
            if use_subcenter:
                self.metric_classify = ArcMarginProductSubcenter(512,
                                           num_classes,
                                           s=s,
                                           m=m,
                                           k=k)
            else:
                self.metric_classify = ArcMarginProduct(512,
                                           num_classes,
                                           s=s,
                                           m=m,
                                           easy_margin=False)

    def forward(self, images, labels=None):
        features = self.model(images)
        features = [
            self.pooling(features[:,:512,:,:]).flatten(1),
            self.pooling(features[:,512:1024,:,:]).flatten(1),
            self.pooling(features[:,1024:1536,:,:]).flatten(1),
            self.pooling(features[:,1536:,:,:]).flatten(1),
        ]
        if self.use_dropout:
            features = [
                self.dropout(features[0]),
                self.dropout(features[1]),
                self.dropout(features[2]),
                self.dropout(features[3]),
            ]

        features = [
            self.fc0(self.dropout(features[0])),
            self.fc1(self.dropout(features[1])),
            self.fc2(self.dropout(features[2])),
            self.fc3(self.dropout(features[3])),
        ]

        features = torch.concat(features)
        if self.out_cosine:
            output = self.metric_classify(features)
        else:
            output = self.metric_classify(features, labels)

        if self.training:
            return output
        else:
            return output, features

class WithCosface(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=1, s=30, m=0.5):
        super(WithCosface, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
        # assert('swin' in model_name)
        if 'swin' in model_name:
            in_features = self.model.head.in_features
            sys.path.append('/home/acc12347av/ml_pipeline/src')
            import timm055
            self.model = timm055.create_model(model_name, pretrained=pretrained, num_classes=1, arcface=True, skip_attn=True, img_size=(768,768))
            self.model.avgpool = nn.Identity()
            self.model.head = nn.Identity()
            self.pooling = nn.AdaptiveAvgPool1d(1)
        elif 'beit' in model_name:
            in_features = self.model.head.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.avgpool = nn.Identity()
            self.model.head = nn.Identity()
            self.model.fc_norm = nn.Identity()
            self.pooling = nn.Identity()
        elif 'convnext' in model_name:
            in_features = self.model.head.fc.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.head = nn.Identity()
            self.pooling = GaP()
        else:
            in_features = self.model.classifier.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.classifier = nn.Identity()
            self.model.global_pool = nn.Identity()
            self.pooling = GaP()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(in_features, 512)
        # print('infeatures:', in_features)
        self.metric_classify = AddMarginProduct(512,
                                   num_classes,
                                   s=s,
                                   m=m)

    def forward(self, images, labels=None):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        # pooled_features = self.dropout(self.pooling(features).flatten(1))
        pooled_features = self.fc(pooled_features)
        output = self.metric_classify(pooled_features, labels)

        if self.training:
            return output
        else:
            return output, pooled_features
