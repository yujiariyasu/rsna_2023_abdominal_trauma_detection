from torch import nn
import torch.nn.functional as F
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class Residual3DBlock(nn.Module):
    def __init__(self, size=512):
        super(Residual3DBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv3d(size, size, 3, stride=1, padding=1),
            nn.BatchNorm3d(size),
            nn.ReLU(size)
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(size, size, 3, stride=1, padding=1),
            nn.BatchNorm3d(size),
        )

    def forward(self, images):
        short_cut = images
        h = self.block(images)
        h = self.block2(h)

        return F.relu(h + short_cut)

# https://www.kaggle.com/competitions/nfl-player-contact-detection/discussion/392402
class TerekaModel(nn.Module):
    def __init__(self, model_name="tf_efficientnet_b0_ns", pretrained=True, num_classes=1, in_chans=3):
        super(TerekaModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes, in_chans=in_chans)
        self.mlp = nn.Sequential(
            nn.Linear(68, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )
        n_hidden = 1024

        self.conv_proj = nn.Sequential(
            nn.Conv2d(1280, 512, 1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.neck = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        self.triple_layer = nn.Sequential(
            Residual3DBlock(size=512),
        )

        self.pool = GeM()

        self.fc = nn.Linear(256+1024, 1)

    def forward(self, images, feature, target=None, mixup_hidden=False, mixup_alpha=0.1, layer_mix=None):
        b, t, h, w = images.shape # 8, 30, 512, 512
        images = images.view(b * t // 3, 3, h, w) # 80, 3, 512, 512
        feature_maps = self.backbone.forward_features(images) # 80, 1280, 16, 16
        feature_maps = self.conv_proj(feature_maps) # 80, 512, 16, 16
        _, c, h, w = feature_maps.size()
        feature_maps = feature_maps.contiguous().view(b*2, c, t//2//3, h, w) # 16, 512, 5, 16, 16
#         import pdb;pdb.set_trace()
        feature_maps = self.triple_layer(feature_maps) # 16, 512, 5, 16, 16
        middle_maps = feature_maps[:, :, 2, :, :] # 16, 512, 16, 16
        middle_maps = self.pool(middle_maps) # 16, 512, 1, 1
        middle_maps = middle_maps.reshape(b, -1) # 8, 1024
        nn_feature = self.neck(middle_maps) # 8, 1024
        feature = self.mlp(feature) # 8, 256
        cat_features = torch.cat([nn_feature, feature], dim=1) # 8, 1280

        if target is not None:
            cat_features, y_a, y_b, lam = mixup_data(cat_features, target, mixup_alpha)
            y = self.fc(cat_features)
            return y, y_a, y_b, lam
        else:
            y = self.fc(cat_features)
            return y

# https://www.kaggle.com/competitions/nfl-player-contact-detection/discussion/391740
class PsiModel(nn.Module):
    def __init__(self, model_name="tf_efficientnet_b0_ns", pretrained=True, num_classes=1, in_chans=3):
        super(PsiModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes, in_chans=in_chans)
        self.mlp = nn.Sequential(
            nn.Linear(68, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )
        n_hidden = 1024

        self.conv_proj = nn.Sequential(
            nn.Conv2d(1280, 1024, 1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.neck = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        self.triple_layer = nn.Sequential(
            Residual3DBlock(size=1024),
        )

        self.pool = GeM()

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, images):
        b, t, h, w = images.shape # 8, 25, 512, 512
        images = images.view(b * t // 5, 5, h, w) # 40, 5, 512, 512
        feature_maps = self.backbone.forward_features(images) # 40, 1280, 16, 16
        feature_maps = self.conv_proj(feature_maps) # 40, 1024, 16, 16
        _, c, h, w = feature_maps.size()
        feature_maps = feature_maps.contiguous().view(b, c, 5, h, w) # 8, 1024, 5, 16, 16
#         import pdb;pdb.set_trace()
        feature_maps = self.triple_layer(feature_maps) # 8, 1024, 5, 16, 16
        feature_maps = feature_maps[:, :, 2, :, :] # 8, 1024, 16, 16
        feature_maps = self.pool(feature_maps) # 8, 1024, 1, 1
        feature_maps = feature_maps.reshape(b, -1) # 8, 1024
        feature_maps = self.neck(feature_maps) # 8, 1024
        y = self.fc(feature_maps)
        return y
