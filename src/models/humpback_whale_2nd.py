import torch
import torch.nn as nn
import timm
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def l2_norm(input, axis=1):
    norm = torch.norm(input,2, axis, True)
    output = torch.div(input, norm)
    return output

def euclidean_dist(x, y):

    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def hard_example_mining(dist_mat, labels, return_inds=False):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    labels = labels.float()
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max((dist_mat * is_pos.float()).contiguous().view(N, -1), 1, keepdim=True)

    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    temp = dist_mat * is_neg.float()
    temp[temp == 0] = 10e5
    dist_an, relative_n_inds = torch.min((temp).contiguous().view(N, -1), 1, keepdim=True)

    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin

        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels):


        global_feat = l2_norm(global_feat)

        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss

def softmax_loss(results, labels):
    labels = labels.view(-1)
    loss = F.cross_entropy(results, labels, reduce=True)
    return loss

def focal_loss(input, target, OHEM_percent=None):
    gamma = 2
    assert target.size() == input.size()

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    invprobs = F.logsigmoid(-input * (target * 2 - 1))
    loss = (invprobs * gamma).exp() * loss

    if OHEM_percent is None:
        return loss.mean()
    else:
        OHEM, _ = loss.topk(k=int(15587 * OHEM_percent), dim=1, largest=True, sorted=True)
        return OHEM.mean()

def bce_loss(input, target, OHEM_percent=None):
    if OHEM_percent is None:
        loss = F.binary_cross_entropy_with_logits(input, target, reduce=True)
        return loss
    else:
        loss = F.binary_cross_entropy_with_logits(input, target, reduce=False)
        value, index= loss.topk(int(15587 * OHEM_percent), dim=1, largest=True, sorted=True)
        return value.mean()

def focal_OHEM(results, labels, labels_onehot, OHEM_percent=100):
    batch_size, class_num = results.shape
    labels = labels.view(-1)
    loss0 = bce_loss(results, labels_onehot, OHEM_percent)
    loss1 = focal_loss(results, labels_onehot, OHEM_percent)
    indexs_ = (labels != class_num).nonzero().view(-1)
    if len(indexs_) == 0:
        return loss0 + loss1
    results_ = results[torch.arange(0,len(results))[indexs_],labels[indexs_]].contiguous()
    loss2 = focal_loss(results_, torch.ones_like(results_).float().cuda())
    return loss0 + loss1 + loss2
def l2_norm(input, axis=1):
    norm = torch.norm(input,2, axis, True)
    output = torch.div(input, norm)
    return output

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

class Prev2ndLoss(object):
    def __init__(self, focal_w=1.0, softmax_w=0.1, triplet_w=100.0):
        self.focal_w = focal_w
        self.softmax_w = softmax_w
        self.triplet_w = triplet_w

    def __call__(self, output, labels):
        labels = labels.long()
        if type(output) != list:
            criterion = nn.CrossEntropyLoss()
            return criterion(output,labels)
        logit, logit_softmax, feas = output
        truth = torch.FloatTensor(len(labels), logit.size(1)+1).to('cuda')
        truth.zero_()
        truth.scatter_(1,labels.view(-1,1),1)
        truth = truth[:, :logit.size(1)]

        truth = to_var(truth)
        labels = to_var(labels)

        loss_focal = focal_OHEM(logit, labels, truth, 1e-2)* self.focal_w
        loss_softmax = softmax_loss(logit_softmax, labels) * self.softmax_w
        loss_triplet = TripletLoss(margin=0.3)(feas, labels) * self.triplet_w
        # print('loss_focal, loss_softmax, loss_triplet:', loss_focal, loss_softmax, loss_triplet)
        return loss_focal + loss_softmax + loss_triplet

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


class MagrginLinear(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=15587,  s=64., m=0.5):
        super(MagrginLinear, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)


    def forward(self, embbedings, label, is_train = False):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m

        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask].float()
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)

        if is_train:
            output[idx_, label.long()] = cos_theta_m[idx_, label.long()].half()

        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

class BinaryHead(nn.Module):

    def __init__(self, num_class=15587, emb_size = 2048, s = 16.0):
        super(BinaryHead,self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea)*self.s
        return logit

###########################################################################################3
class MarginHead(nn.Module):

    def __init__(self, num_class=15587, emb_size = 2048, s=64., m=0.5):
        super(MarginHead,self).__init__()
        self.fc = MagrginLinear(embedding_size=emb_size, classnum=num_class , s=s, m=m)

    def forward(self, fea, label, is_train):
        fea = l2_norm(fea)
        logit = self.fc(fea, label, is_train)
        return logit

###########################################################################################3
class WhalePrev2ndModel(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=1, s=30, m=0.5, k=3, use_gem=True, use_dropout=False, use_subcenter=True, out_cosine=True):
        super(WhalePrev2ndModel, self).__init__()
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

        self.fea_bn = nn.BatchNorm1d(in_features)
        self.fea_bn.bias.requires_grad_(False)
        self.margin_head = MarginHead(num_classes, emb_size=in_features, s = s, m = m)
        self.binary_head = BinaryHead(num_classes, emb_size=in_features, s = s)

    def forward(self, x, label=None):
        fea = self.model(x)
        fea = self.pooling(fea)
        fea = fea.view(fea.size(0), -1)
        fea = self.fea_bn(fea)
        logit_binary = self.binary_head(fea)
        logit_margin = self.margin_head(fea, label = label, is_train = self.training)
        if self.training:
            return [logit_binary, logit_margin, fea]
        else:
            return logit_binary, fea

class Net(nn.Module):
    def __init__(self, num_class=None, s1 = 64 , m1 = 0.5, s2 = 64):
        super(Net,self).__init__()
        self.s1 = s1
        self.m1 = m1
        self.s2 = s2

        self.basemodel = se_resnet101()
        self.basemodel.avg_pool =  nn.AdaptiveAvgPool2d(1)
        self.basemodel.last_linear = nn.Sequential()

        emb_size = 2048
        self.fea_bn = nn.BatchNorm1d(emb_size)
        self.fea_bn.bias.requires_grad_(False)

        self.margin_head = MarginHead(num_class, emb_size=emb_size, s = self.s1, m = self.m1)
        self.binary_head = BinaryHead(num_class, emb_size=emb_size, s = self.s2)

    def forward(self, x, label = None, is_infer = None):
        mean = [0.485, 0.456, 0.406]  # rgb
        std = [0.229, 0.224, 0.225]

        x = torch.cat([
            (x[:, [0]] - mean[0]) / std[0],
            (x[:, [1]] - mean[1]) / std[1],
            (x[:, [2]] - mean[2]) / std[2],
        ], 1)

        x = self.basemodel.layer0(x)
        x = self.basemodel.layer1(x)
        x = self.basemodel.layer2(x)
        x = self.basemodel.layer3(x)
        x = self.basemodel.layer4(x)

        x = self.basemodel.avg_pool(x)
        fea = x.view(x.size(0), -1)
        fea = self.fea_bn(fea)
        logit_binary = self.binary_head(fea)
        logit_margin = self.margin_head(fea, label = label, is_train = self.training)

        return logit_binary, logit_margin, fea
        ###
        loss_focal = focal_OHEM(logit, truth_,truth, hard_ratio)* config.focal_w
        loss_softmax = softmax_loss(logit_softmax[indexs_NoNew], truth_[indexs_NoNew]) * config.softmax_w
        loss_triplet = TripletLoss(margin=0.3)(feas, truth_) * config.triplet_w
        loss = loss_focal + loss_softmax + loss_triplet

        ###
