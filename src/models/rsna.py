from .senet import se_resnext50_32x4d
from .backbones import senet_mod

import torch
import torch.nn as nn
import timm
from pdb import set_trace as st
import torch.nn.functional as F
import math
from pdb import set_trace as st

class RsnaModel(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=1, meta_cols_num=1):
        super(RsnaModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
        if 'swin' in model_name:
            in_features = self.model.head.in_features
            self.model.head = nn.Identity()
            self.fc = nn.Linear(in_features+meta_cols_num, num_classes)
        elif ('resnet' in model_name) | ('resnext' in model_name):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
            self.fc = nn.Linear(in_features+meta_cols_num, num_classes)
        elif 'efficientnet' in model_name:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
            self.fc = nn.Linear(in_features+meta_cols_num, num_classes)
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
            self.model.head = nn.Identity()
            self.model.fc_norm = nn.Identity()
            self.pooling = nn.Identity()
            self.fc = nn.Linear(in_features+meta_cols_num, num_classes)
        elif 'convnext' in model_name:
            in_features = self.model.head.fc.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.head.fc = nn.Identity()
            self.fc = nn.Linear(in_features+meta_cols_num, num_classes)
        else:
            in_features = self.model.classifier.in_features
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
            self.model.classifier = nn.Identity()
            self.model.global_pool = nn.Identity()
            if use_gem:
                self.pooling = GeM()
            else:
                self.pooling = GaP()

    def forward(self, input_):
        images, slice_n = input_
        features = self.model(images)
        # import pdb;pdb.set_trace()
        features = torch.cat([features, slice_n], 1)
        return self.fc(features)

    def extract_features(self, input_):
        images, slice_n = input_
        return self.model(images)

    def extract_with_features(self, input_):
        images, slice_n = input_
        features = self.model(images)
        # import pdb;pdb.set_trace()
        features = torch.cat([features, slice_n], 1)
        return self.fc(features), features



class RSNAStage2(nn.Module):
    def __init__(self):
        super(RSNAStage2, self).__init__()
        num_feature = 1536+8
        num_classes = 8
        # self.rnn1 = nn.LSTM(
        #     num_feature,
        #     512,
        #     num_layers=2,
        #     batch_first=True,
        #     bidirectional=True,
        # )
        # self.deconv1 = nn.ConvTranspose1d(
        #     1024, 512, kernel_size=3, stride=2, padding=1
        # )
        # self.rnn2 = nn.LSTM(
        #     1024,
        #     512,
        #     num_layers=2,
        #     batch_first=True,
        #     bidirectional=True,
        # )
        # self.deconv2 = nn.ConvTranspose1d(
        #     1024, 512, kernel_size=3, stride=2, padding=1
        # )
        # out_features = 1024
        self.rnn1 = nn.GRU(
            num_feature,
            512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.deconv1 = nn.ConvTranspose1d(
            1024, 512, kernel_size=3, stride=2, padding=1
        )
        self.rnn2 = nn.GRU(
            1024,
            512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.deconv2 = nn.ConvTranspose1d(
            1024, 512, kernel_size=3, stride=2, padding=1
        )
        out_features = 1024
        self.exam_classfifier = nn.Linear(out_features, num_classes)
        self.image_classfifier = nn.Linear(out_features, 1)

    def pooling(self, feature, seq_lens):
        pool_out = torch.stack(
            [feature[idx, :seq_len].mean(0) for idx, seq_len in enumerate(seq_lens)]
        )
        return pool_out

    def forward(self, feature):
        outs = []
        feature, _ = self.rnn1(feature)
        out = self.deconv1(feature.permute(0, 2, 1)).permute(0, 2, 1)
        outs.append(out)
        feature, _ = self.rnn2(feature)
        out = self.deconv2(feature.permute(0, 2, 1)).permute(0, 2, 1)
        outs.append(out)
        feature = torch.cat(outs, dim=-1)
        pool_out = self.pooling(feature, range(1))
        exam_out = self.exam_classfifier(pool_out)
        return exam_out


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class RSNA_CNN_1D(nn.Module):
    def __init__(self, input_ch=1536+8, num_classes=8):
        super(RSNA_CNN_1D, self).__init__()
        pool = 4
        drop = 0.1
        print(input_ch)
        self.layer1 = nn.Sequential(
                nn.Conv1d(input_ch//pool, 64, kernel_size=7, stride=1, padding=3, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                SEModule(64, 16),
#                 nn.Dropout(drop),
        )
        self.fpool = nn.MaxPool1d(kernel_size=pool, stride=pool, padding=0)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer2 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                SEModule(128, 16),
#                 nn.Dropout(drop),
        )
        self.layer3 = nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                SEModule(256, 16),
#                 nn.Dropout(drop),
        )
        self.layer4 = nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                SEModule(512, 16),
#                 nn.Dropout(drop),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc2 = nn.Conv1d(
        #     input_ch//pool+64+128+256+512,
        #     2, kernel_size=1)
#         self.fc = nn.Linear(512, 8)
        self.fc = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
        )

    def forward(self, x_input):
        bs, ch, d = x_input.size()
        x0 = torch.transpose(x_input, 1, 2)
        x0 = self.fpool(x0)
        x0 = torch.transpose(x0, 1, 2)
        x1 = self.layer1(x0)
        x1 = self.maxpool(x1)

        x2 = self.layer2(x1)
        x2 = self.maxpool(x2)
        x3 = self.layer3(x2)
        x3 = self.maxpool(x3)
        x4 = self.layer4(x3)

#         tmp = F.adaptive_avg_pool1d(x1, d)
#         print(tmp.shape)
#         tmp = F.adaptive_avg_pool1d(x2, d)
#         print(tmp.shape)
        x5 = torch.cat([
            x0,
            F.adaptive_avg_pool1d(x1, d),
            F.adaptive_avg_pool1d(x2, d),
            F.adaptive_avg_pool1d(x3, d),
            F.adaptive_avg_pool1d(x4, d),
        ], axis=1)
        # y2 = self.fc2(x5)
        b, ch, d = x_input.size()
#         x1 = self.fc(x)
#         x1 = x1.view(b, -1, 1)
        y = self.avgpool(x4)
        y = y.view(b, -1)
        y = self.fc(y)
        return y



class ResNet(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, dropout=0.0):
        super(ResNet, self).__init__()
        assert kernel_size % 2 == 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_features,
                out_features,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                out_features,
                out_features,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x) + x
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=201):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, : d_model // 2]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)



class DeconvFeatureModel(nn.Module):
    def __init__(self, num_feature, num_classes, backbone, dropout_rate=0.3):
        super(DeconvFeatureModel, self).__init__()
        self.backbone = backbone
        # self.rnn1 = nn.GRU(
        #     num_feature,
        #     512,
        #     num_layers=2,
        #     batch_first=True,
        #     bidirectional=True,
        # )
        # self.deconv1 = nn.ConvTranspose1d(
        #     1024, 512, kernel_size=3, stride=2, padding=1
        # )
        # self.rnn2 = nn.GRU(
        #     1024,
        #     512,
        #     num_layers=2,
        #     batch_first=True,
        #     bidirectional=True,
        # )
        # self.deconv2 = nn.ConvTranspose1d(
        #     1024, 512, kernel_size=3, stride=2, padding=1
        # )
        # out_features = 1024
        if backbone in ["cnn", "cnn_rnn"]:
            self.resnet1 = ResNet(
                num_feature, 512, kernel_size=3, dropout=dropout_rate
            )
            self.deconv1 = nn.ConvTranspose1d(
                512, 512, kernel_size=3, stride=2, padding=1
            )
            self.resnet2 = ResNet(
                512, 256, kernel_size=5, dropout=dropout_rate
            )
            self.deconv2 = nn.ConvTranspose1d(
                256, 256, kernel_size=3, stride=2, padding=1
            )
            self.resnet3 = ResNet(
                256, 128, kernel_size=7, dropout=dropout_rate
            )
            self.deconv3 = nn.ConvTranspose1d(
                128, 128, kernel_size=3, stride=2, padding=1
            )
            self.resnet4 = ResNet(
                128, 64, kernel_size=9, dropout=dropout_rate
            )
            self.deconv4 = nn.ConvTranspose1d(
                64, 64, kernel_size=3, stride=2, padding=1
            )
            self.resnet5 = ResNet(
                64, 32, kernel_size=11, dropout=dropout_rate
            )
            self.deconv5 = nn.ConvTranspose1d(
                32, 32, kernel_size=3, stride=2, padding=1
            )
            out_features = 512 + 256 + 128 + 64 + 32
        elif backbone == "lstm":
            self.rnn1 = nn.LSTM(
                num_feature,
                512,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            self.deconv1 = nn.ConvTranspose1d(
                1024, 512, kernel_size=3, stride=2, padding=1
            )
            self.rnn2 = nn.LSTM(
                1024,
                512,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            self.deconv2 = nn.ConvTranspose1d(
                1024, 512, kernel_size=3, stride=2, padding=1
            )
            out_features = 1024
        elif backbone == "gru":
            self.rnn1 = nn.GRU(
                num_feature,
                512,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            self.deconv1 = nn.ConvTranspose1d(
                1024, 512, kernel_size=3, stride=2, padding=1
            )
            self.rnn2 = nn.GRU(
                1024,
                512,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            self.deconv2 = nn.ConvTranspose1d(
                1024, 512, kernel_size=3, stride=2, padding=1
            )
            out_features = 1024
        elif backbone in ["transformer", "transformer_rnn"]:
            self.linear = nn.Linear(num_feature, 2048)
            # self.scale = math.sqrt(2048)
            self.pe = PositionalEncoding(2048, dropout_rate)
            encoder_layer1 = nn.TransformerEncoderLayer(
                2048,
                nhead=8,
                dim_feedforward=1024,
                dropout=dropout_rate,
                activation="gelu",
            )
            self.transformer1 = nn.TransformerEncoder(encoder_layer1, 1)
            self.deconv1 = nn.ConvTranspose1d(
                2048, 1024, kernel_size=3, stride=2, padding=1
            )
            encoder_layer2 = nn.TransformerEncoderLayer(
                1024,
                nhead=8,
                dim_feedforward=1024,
                dropout=dropout_rate,
                activation="gelu",
            )
            # self.transformer2 = nn.TransformerEncoder(encoder_layer2, 1)
            self.deconv2 = nn.ConvTranspose1d(
                2048, 1024, kernel_size=3, stride=2, padding=1
            )
            out_features = 2048
        else:
            raise NotImplementedError()
        if backbone in ["cnn_rnn", "transformer_rnn"]:
            self.rnn = nn.LSTM(
                out_features,
                512,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            out_features = 1024
        self.exam_classfifier = nn.Linear(out_features, num_classes)
        # self.image_classfifier = nn.Linear(out_features, 1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def pooling(self, feature, seq_lens):
        pool_out = torch.stack(
            [feature[idx, :seq_len].mean(0) for idx, seq_len in enumerate(seq_lens)]
        )
        # pool_out = feature.mean(1)
        return pool_out

    def forward(self, feature):
        seq_len = feature.size(1)
        # outs = []
        # feature, _ = self.rnn1(feature)
        # out = self.deconv1(feature.permute(0, 2, 1)).permute(0, 2, 1)
        # outs.append(out)
        # feature, _ = self.rnn2(feature)
        # out = self.deconv2(feature.permute(0, 2, 1)).permute(0, 2, 1)
        # outs.append(out)
        # feature = torch.cat(outs, dim=-1)
        # print('feature.size():',feature.size())
        if self.backbone in ["cnn", "cnn_rnn"]:
            feature = feature.permute(0, 2, 1)
            outs = []
            for i in range(1, 6):
                feature = getattr(self, f"resnet{i}")(feature)
                out = getattr(self, f"deconv{i}")(feature)
                outs.append(out)
            feature = torch.cat(outs, dim=1).permute(0, 2, 1)
        elif self.backbone in ["gru", "lstm"]:
            outs = []
            feature, _ = self.rnn1(feature)
            out = self.deconv1(feature.permute(0, 2, 1)).permute(0, 2, 1)
            outs.append(out)
            feature, _ = self.rnn2(feature)
            out = self.deconv2(feature.permute(0, 2, 1)).permute(0, 2, 1)
            outs.append(out)
            feature = torch.cat(outs, dim=-1)
        elif self.backbone in ["transformer", "transformer_rnn"]:
            # st()
            feature = torch.relu(self.linear(feature))
            feature = self.pe(feature.permute(1, 0, 2))
            outs = []
            feature = self.transformer1(feature)
            out = self.deconv1(feature.permute(1, 2, 0)).permute(2, 0, 1)
            outs.append(out)
            feature = self.transformer1(feature)
            out = self.deconv2(feature.permute(1, 2, 0)).permute(2, 0, 1)
            outs.append(out)
            feature = torch.cat(outs, dim=-1).permute(1, 0, 2)
        else:
            pass
        if self.backbone in ["cnn_rnn", "transformer_rnn"]:
            feature, _ = self.rnn(feature)
        # feature, _ = self.rnn(feature)
        # pool_out = self.pooling(feature, [seq_len])
        pool_out = torch.mean(feature, axis=1)
        # pool_out = self.avgpool(feature)
        output = self.exam_classfifier(pool_out)
        # print(output)
        return output


class RSNAMultiImageModel(nn.Module):
    def __init__(self, encoder, image_size, image_n, pretrained=False, num_classes=1, in_chans=3, in_features=1024):
        super(MultiImageModel, self).__init__()
        self.encoder = encoder
        self.image_size = image_size
        self.image_n = image_n
        self.in_chans = in_chans
        # self.encoder.head.fc = nn.Linear(in_features, num_classes)
        self.lstm = nn.LSTM(in_features, 256, num_layers=2, dropout=0, bidirectional=True, batch_first=True)

        feat_n = image_n * 512
        # self.head_age = nn.Sequential(nn.Linear(feat_n, 1))
        # self.head_l = nn.Sequential(nn.Linear(feat_n, 2))
        # self.head_r = nn.Sequential(nn.Linear(feat_n, 2))

        # self.head_age = nn.Sequential(nn.Linear(feat_n, 1))
        # self.head_r_cancer = nn.Sequential(nn.Linear(feat_n, 1))
        # self.head_l_cancer = nn.Sequential(nn.Linear(feat_n, 1))
        # self.head_r_biopsy = nn.Sequential(nn.Linear(feat_n, 1))
        # self.head_l_biopsy = nn.Sequential(nn.Linear(feat_n, 1))


        self.head_l = nn.Sequential(
            nn.Linear(feat_n, 1024),
            # nn.BatchNorm1d(1024),
            nn.Dropout(0),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 3),
        )
        self.head_r = nn.Sequential(
            nn.Linear(feat_n, 1024),
            # nn.BatchNorm1d(1024),
            nn.Dropout(0),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 3),
        )
        self.head_age = nn.Sequential(
            nn.Linear(feat_n, 1024),
            # nn.BatchNorm1d(1024),
            nn.Dropout(0),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        bs = x.shape[0] # (bs, image_n, 3, 1024, 512)
        # print('x.shape:', x.shape)
        x = x.view(bs * self.image_n, self.in_chans, self.image_size[0], self.image_size[1]) # (bs*image_n, 3, 1024, 512)
        # st()
        # x = x[:,0,:,:,:]
        # return self.encoder(x) # (bs*image_n, in_features)
        feat = self.encoder(x) # (bs*image_n, in_features)
        feat = feat.view(bs, self.image_n, -1)
        feat, _ = self.lstm(feat) # (bs, image_n, 512)

        # st()

        feat = feat.contiguous().view(bs, -1) # (bs, image_n*512)
        # feat = feat.view(bs, -1) # (bs, image_n*512)
        # st()
        l = self.head_l(feat)
        r = self.head_r(feat)
        age = self.head_age(feat)
        return torch.cat([l, r, age], axis=1) # (bs, num_classes)


        # l_cancer = self.head_l_cancer(feat)
        # r_cancer = self.head_r_cancer(feat)
        # l_biopsy = self.head_l_biopsy(feat)
        # r_biopsy = self.head_r_biopsy(feat)
        # age = self.head_age(feat)
        # return torch.cat([l_cancer, l_biopsy, r_cancer, r_biopsy, age], axis=1) # (bs, num_classes)



        # feat = feat.contiguous().view(bs, -1) # (bs, image_n*in_features)

        # age = self.head_age(feat)
        # return self.head_l(feat)
        # feat_l = self.head_l(feat)
        # feat_r = self.head_r(feat)
        # return torch.cat([feat_l, feat_r], axis=1) # (bs, num_classes)

class MultiImageModel(nn.Module):
    def __init__(self, encoder, image_size, n_instance, pretrained=False, num_classes=1, in_chans=3, in_features=1024):
        super(MultiImageModel, self).__init__()
        self.encoder = encoder
        self.image_size = image_size
        self.n_instance = n_instance
        self.in_chans = in_chans
        self.lstm = nn.LSTM(in_features, 256, num_layers=2, dropout=0, bidirectional=True, batch_first=True)

        feat_n = image_n * 512

        self.head = nn.Sequential(
            nn.Linear(feat_n, 1024),
            # nn.BatchNorm1d(1024),
            nn.Dropout(0),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 3),
        )

    def forward(self, x):
        bs = x.shape[0] # (bs, n_instance, ch, 512, 512)
        x = x.view(bs * self.n_instance, self.in_chans, self.image_size[0], self.image_size[1]) # (bs*n_instance, ch, 512, 512)
        # st()
        feat = self.encoder(x) # (bs*n_instance, in_features)
        feat = feat.view(bs, self.n_instance, -1)
        feat, _ = self.lstm(feat) # (bs, n_instance, 512)

        feat = feat.contiguous().view(bs, -1) # (bs, n_instance*512)
        return self.head(feat)

class Timm1BoneModel(nn.Module):
    def __init__(self, backbone, n_instance=5, image_size=384, pretrained=True, in_chans=3, output_1=True, use_final=False, num_classes=1):
        super(Timm1BoneModel, self).__init__()
        self.image_size = image_size
        self.in_chans = in_chans
        self.output_1 = output_1
        self.n_instance = n_instance
        self.num_classes = num_classes

        if backbone == 'se_resnext50_32x4d':
            self.encoder = senet_mod(se_resnext50_32x4d, pretrained=True)
        else:
            self.encoder = timm.create_model(
                backbone,
                in_chans=in_chans,
                num_classes=1,
                features_only=False,
                drop_rate=0,
                drop_path_rate=0,
                pretrained=pretrained
            )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif ('convnext' in backbone) or ('nfnet' in backbone):
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()
        else:
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
            hdim = list(self.encoder.children())[-1].in_features

        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=0, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),
            nn.Linear(256, num_classes),
        )
        # self.head = nn.Sequential(
        #     nn.Linear(512*n_instance, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.Dropout(0.3),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(1024, num_classes),
        # )
        self.use_final = use_final


    def forward(self, x):  # (bs, 15, 6, 224, 224)
        bs = x.shape[0]
        # print('bs:', bs)
        x = x.view(bs * self.n_instance, self.in_chans, self.image_size, self.image_size) # (bs*15, 6, 224, 224)
        feat = self.encoder(x)
        feat = feat.view(bs, self.n_instance, -1)
        feat, _ = self.lstm(feat) # (bs, 15, 512)
        # print(feat.size())
        if self.output_1:
            if self.use_final:
                feat = feat[:, -1] # (bs, 512)
            else:
                feat = torch.mean(feat, 1)
                # feat = feat.reshape(bs, -1)

            return self.head(feat) # (bs, num_classes)
        else:
            feat = feat.contiguous().view(bs * self.n_instance, -1) # (bs*15, 512)
            feat = self.head(feat) # (bs*15, num_classes)
            feat = feat.view(bs, self.num_classes*self.n_instance).contiguous() # (bs, num_classes*15)
            return feat


import torch.nn as nn
from itertools import repeat

class SpatialDropout(nn.Module):
    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop
        
    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim()-2), inputs.shape[-1]) 
        
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)    
            outputs.mul_(noises)
            return outputs
            
    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)


import torch
from torch import nn
import torch.nn.functional as F

from typing import Dict, Optional
 
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


    
class MLPAttentionNetwork(nn.Module):
 
    def __init__(self, hidden_dim, attention_dim=None):
        super(MLPAttentionNetwork, self).__init__()
 
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        if self.attention_dim is None:
            self.attention_dim = self.hidden_dim
        # W * x + b
        self.proj_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=True)
        # v.T
        self.proj_v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, x):
        """
        :param x: seq_len, batch_size, hidden_dim
        :return: batch_size * seq_len, batch_size * hidden_dim
        """
        # print(f"x shape:{x.shape}")
        batch_size, seq_len, _ = x.size()
        # flat_inputs = x.reshape(-1, self.hidden_dim) # (batch_size*seq_len, hidden_dim)
        # print(f"flat_inputs shape:{flat_inputs.shape}")

        H = torch.tanh(self.proj_w(x)) # (batch_size, seq_len, hidden_dim)
        # print(f"H shape:{H.shape}")

        att_scores = torch.softmax(self.proj_v(H),axis=1) # (batch_size, seq_len)
        # print(f"att_scores shape:{att_scores.shape}")

        attn_x = (x * att_scores).sum(1) # (batch_size, hidden_dim)
        # print(f"attn_x shape:{attn_x.shape}")
        return attn_x

class RSNAClassifier2nd(nn.Module):
    def __init__(self, model_arch, hidden_dim=256, seq_len=24, pretrained=False, dropout=0.1, num_classes=1):
        super().__init__()
        self.seq_len = seq_len
        self.model = timm.create_model(model_arch, in_chans=1, pretrained=False)
        self.model_arch = model_arch

        if 'efficientnet' in self.model_arch:
            cnn_feature = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif "res" in self.model_arch:
            cnn_feature = self.model.fc.in_features
            self.model.global_pool = nn.Identity()
            self.model.fc = nn.Identity()
            self.pooling = nn.AdaptiveAvgPool2d(1)

        self.spatialdropout = SpatialDropout(dropout)
        self.gru = nn.GRU(cnn_feature, hidden_dim, 2, batch_first=True, bidirectional=True)
        self.mlp_attention_layer = MLPAttentionNetwork(2 * hidden_dim)
        self.logits = nn.Sequential(
            nn.Linear(hidden_dim*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # for n, m in self.named_modules():
        #     if isinstance(m, nn.GRU):
        #         print(f"init {m}")
        #         for param in m.parameters():
        #             if len(param.shape) >= 2:
        #                 nn.init.orthogonal_(param.data)
        #             else:
        #                 nn.init.normal_(param.data)

    def forward(self, x): # (B, seq_len, H, W)
        bs = x.size(0)
        x = x.reshape(bs*self.seq_len, 1, x.size(2), x.size(3)) # (B*seq_len, 1, H, W)
        features = self.model(x)
        if "res" in self.model_arch:
            features = self.pooling(features).view(bs*self.seq_len, -1) # (B*seq_len, cnn_feature)
        features = self.spatialdropout(features)                # (B*seq_len, cnn_feature)
        # print(features.shape)
        features = features.reshape(bs, self.seq_len, -1)       # (B, seq_len, cnn_feature)
        features, _ = self.gru(features)                        # (B, seq_len, hidden_dim*2)
        atten_out = self.mlp_attention_layer(features)          # (B, hidden_dim*2)
        pred = self.logits(atten_out)                           # (B, 1)
        pred = pred.view(bs, -1)                                # (B, 1)
        return pred



def drop_fc(model):
    if model.__class__.__name__ in ['Xcit', 'Cait']:
        nc = model.head.in_features
        model.norm = nn.Identity()
        model.head_drop = nn.Identity()
        model.head = nn.Identity()
        new_model = model
    elif model.__class__.__name__ == 'RegNet':
        nc = model.head.fc.in_features
        model.head.global_pool = nn.Identity()
        model.head.flatten = nn.Identity()
        model.head.drop = nn.Identity()
        model.head.fc = nn.Identity()
        new_model = model
    elif model.__class__.__name__ == 'MetaFormer':
        nc = model.head.fc.fc1.in_features
        model.head.global_pool = nn.Identity()
        model.head.norm = nn.Identity()
        model.head.flatten = nn.Identity()
        model.head.drop = nn.Identity()
        model.head.fc = nn.Identity()
        new_model = model
    elif model.__class__.__name__ == 'Sequencer2D':
        nc = model.head.in_features
        model.head = nn.Identity()
        new_model = model
    elif model.__class__.__name__ == 'Sam':
        new_model = model.image_encoder
        nc = new_model.neck[0].out_channels
    elif 'SwinTransformer' in model.__class__.__name__:
        nc = model.head.in_features
        model.head = nn.Identity()
        new_model = model
    elif model.__class__.__name__ == 'ConvNeXt':
        nc = model.head.fc.in_features
        model.head.global_pool = nn.Identity()
        model.head.norm = nn.Identity()
        model.head.flatten = nn.Identity()
        model.head.drop = nn.Identity()
        model.head.fc = nn.Identity()
        new_model = model
    elif model.__class__.__name__ == 'FeatureEfficientNet':
        new_model = model
        nc = model._fc.in_features
    elif model.__class__.__name__ == 'RegNetX':
        new_model = nn.Sequential(*list(model.children())[0])[:-1]
        nc = list(model.children())[0][-1].fc.in_features
    elif model.__class__.__name__ == 'DenseNet':
        new_model = nn.Sequential(*list(model.children())[:-1])
        nc = list(model.children())[-1].in_features
    # elif model.__class__.__name__ == 'EfficientNet':
    #     new_model = nn.Sequential(*list(model.children())[:-2])
    #     import pdb;pdb.set_trace()
    #     nc = 1280
    else:
        new_model = nn.Sequential(*list(model.children())[:-2])
        nc = list(model.children())[-1].in_features
    return new_model, nc

class MultiInstanceCNNModelRetrain(nn.Module):
    def __init__(
        self,
        num_classes=1,
        n_instance=5,
        center_loss_feat_dim=512,
        base_model=senet_mod(se_resnext50_32x4d, pretrained=True),
    ):
        super().__init__()
        self.n_instance = n_instance
        self.criterion = nn.CrossEntropyLoss()
        self.center_loss_feat_dim = center_loss_feat_dim
        self.model_name = base_model.__class__.__name__
        self.encoder, nc = drop_fc(base_model)
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.Linear(2 * nc, self.center_loss_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(self.center_loss_feat_dim, num_classes)

    def forward(self, x):
        # st()
        bs, n, ch, w, h = x.shape # 8, 5, 3, 384, 384
        x = x.view(bs * n, ch, w, h) # 40, 3, 384, 384
        x = self.encoder(x) # 40, 2048, 20, 20
        bs2, ch2, w2, h2 = x.shape

        x = (
            x.view(-1, n, ch2, w2, h2)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(bs, ch2, n * w2, h2)
        )  # x: bs x C' x N W'' x W''
        x = self.head(x)
        x = self.fc(x)
        return x

class RSNA2ndModel(nn.Module):
    def __init__(
        self,
        num_classes=1,
        n_instance=5,
        base_model=senet_mod(se_resnext50_32x4d, pretrained=True),
        pool='avg',
    ):
        super().__init__()
        self.n_instance = n_instance
        self.criterion = nn.CrossEntropyLoss()
        self.model_name = base_model.__class__.__name__
        self.encoder, nc = drop_fc(base_model)
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.Linear(2 * nc, num_classes),
        )
    def forward(self, x):
        # st()
        bs, n, ch, w, h = x.shape # 8, 5, 3, 384, 384
        x = x.view(bs * n, ch, w, h) # 40, 3, 384, 384
        x = self.encoder(x) # 40, 2048, 20, 20
        bs2, ch2, w2, h2 = x.shape

        x = (
            x.view(-1, n, ch2, w2, h2)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(bs, ch2, n * w2, h2)
        )  # x: bs x C' x N W'' x W''
        x = self.head(x)
        return x

class RSNA2ndModel2(nn.Module):
    def __init__(
        self,
        num_classes=1,
        n_instance=5,
        center_loss_feat_dim=512,
        base_model=senet_mod(se_resnext50_32x4d, pretrained=True),
        pool='avg'
    ):
        super().__init__()
        self.n_instance = n_instance
        self.criterion = nn.CrossEntropyLoss()
        self.center_loss_feat_dim = center_loss_feat_dim
        self.model_name = base_model.__class__.__name__
        self.encoder, nc = drop_fc(base_model)
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1,1))
        else:
            self.pool = nn.AdaptiveMaxPool2d((1,1))
        self.head = nn.Sequential(
            Flatten(),
            nn.Linear(nc, self.center_loss_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(self.center_loss_feat_dim, num_classes)

    def forward(self, x):
        # st()
        bs, n, ch, w, h = x.shape # 8, 5, 3, 384, 384
        x = x.view(bs * n, ch, w, h) # 40, 3, 384, 384
        x = self.encoder(x) # 40, 2048, 20, 20
        bs2, ch2, w2, h2 = x.shape

        x = (
            x.view(-1, n, ch2, w2, h2)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(bs, ch2, n * w2, h2)
        )  # x: bs x C' x N W'' x W''
        x = self.pool(x)
        x = self.head(x)
        x = self.fc(x)
        return x

class RSNA2ndModelXcit(nn.Module):
    def __init__(
        self,
        num_classes=1,
        n_instance=5,
        center_loss_feat_dim=512,
        base_model=senet_mod(se_resnext50_32x4d, pretrained=True),
        pool='avg'
    ):
        super().__init__()
        self.n_instance = n_instance
        self.criterion = nn.CrossEntropyLoss()
        self.center_loss_feat_dim = center_loss_feat_dim
        self.model_name = base_model.__class__.__name__
        self.encoder, nc = drop_fc(base_model)
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1,1))
        else:
            self.pool = nn.AdaptiveMaxPool2d((1,1))
        self.head = nn.Sequential(
            Flatten(),
            nn.Linear(nc, self.center_loss_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(self.center_loss_feat_dim, num_classes)

    def forward(self, x):
        # st()
        bs, n, ch, w, h = x.shape # 8, 5, 3, 384, 384
        x = x.view(bs * n, ch, w, h) # 40, 3, 384, 384
        x = self.encoder(x) # 40, 2048, 20, 20

        x = (
            x.view(-1, n, ch2, w2, h2)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(bs, ch2, n * w2, h2)
        )  # x: bs x C' x N W'' x W''
        x = self.pool(x)
        x = self.head(x)
        x = self.fc(x)
        return x

class Rsna2boxel(nn.Module):
    def __init__(
        self,
        num_classes=1,
        n_instance=5,
        center_loss_feat_dim=512,
        base_model=senet_mod(se_resnext50_32x4d, pretrained=True),
        base_model2=senet_mod(se_resnext50_32x4d, pretrained=True),
        pool='avg'
    ):
        super().__init__()
        self.n_instance = n_instance
        self.criterion = nn.CrossEntropyLoss()
        self.center_loss_feat_dim = center_loss_feat_dim
        self.model_name = base_model.__class__.__name__
        self.encoder, nc = drop_fc(base_model)
        # self.encoder2, nc = drop_fc(base_model2)
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1,1))
        else:
            self.pool = nn.AdaptiveMaxPool2d((1,1))
        self.head = nn.Sequential(
            Flatten(),
            nn.Linear(nc*2, self.center_loss_feat_dim),
            # nn.Linear(nc, self.center_loss_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(self.center_loss_feat_dim, num_classes)

    def forward(self, boxel2):
        x, x2 = boxel2
        bs, n, ch, w, h = x.shape # 8, 5, 3, 384, 384
        x = x.view(bs * n, ch, w, h) # 40, 3, 384, 384
        x = self.encoder(x) # 40, 2048, 20, 20
        bs2, ch2, w2, h2 = x.shape

        x = (
            x.view(-1, n, ch2, w2, h2)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(bs, ch2, n * w2, h2)
        )  # x: bs x C' x N W'' x W''
        x = self.pool(x)

        x2 = x2.view(bs * n, ch, w, h) # 40, 3, 384, 384
        x2 = self.encoder(x2) # 40, 2048, 20, 20
        x2 = (
            x2.view(-1, n, ch2, w2, h2)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(bs, ch2, n * w2, h2)
        )  # x: bs x C' x N W'' x W''
        x2 = self.pool(x2)
        x = torch.cat([x, x2], 1)
        x = self.head(x)
        x = self.fc(x)
        return x

class RSNA2ndModel3(nn.Module):
    def __init__(
        self,
        num_classes=1,
        n_instance=5,
        base_model=senet_mod(se_resnext50_32x4d, pretrained=True),
        pool='avg'
    ):
        super().__init__()
        self.n_instance = n_instance
        self.criterion = nn.CrossEntropyLoss()
        self.model_name = base_model.__class__.__name__
        self.encoder, nc = drop_fc(base_model)
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1,1))
        else:
            self.pool = nn.AdaptiveMaxPool2d((1,1))

        self.head = nn.Sequential(
            Flatten(),
            nn.Linear(nc, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # st()
        bs, n, ch, w, h = x.shape # 8, 5, 3, 384, 384
        x = x.view(bs * n, ch, w, h) # 40, 3, 384, 384
        x = self.encoder(x) # 40, 2048, 20, 20
        bs2, ch2, w2, h2 = x.shape

        x = (
            x.view(-1, n, ch2, w2, h2)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(bs, ch2, n * w2, h2)
        )  # x: bs x C' x N W'' x W''
        x = self.pool(x)
        x = self.head(x)
        return x

class RsnaLstm(nn.Module):
    def __init__(
        self,
        num_classes=1,
        n_instance=5,
        center_loss_feat_dim=512,
        base_model=senet_mod(se_resnext50_32x4d, pretrained=True, in_channel=3),
    ):
        super().__init__()
        self.n_instance = n_instance
        self.criterion = nn.CrossEntropyLoss()
        self.center_loss_feat_dim = center_loss_feat_dim
        self.model_name = base_model.__class__.__name__
        self.encoder, nc = drop_fc(base_model)
        self.pool = nn.Sequential(
            AdaptiveConcatPool2d(),
        )
        hidden_dim = 256
        self.lstm = nn.LSTM(nc*2, hidden_dim, num_layers=2, dropout=0, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2*n_instance, num_classes)

    def forward(self, x):
        bs, n, ch, w, h = x.shape # 8, 5, 3, 384, 384
        x = x.view(bs * n, ch, w, h) # 40, 3, 384, 384
        x = self.encoder(x) # 40, 2048, 20, 20
        # st()
        x = self.pool(x)
        x = x.view(bs, n, -1)
        x, _ = self.lstm(x)                        # (B, seq_len, hidden_dim*2)
        x = x.reshape(bs, -1)
        x = self.fc(x)
        return x

class RsnaLstmXcit(nn.Module):
    def __init__(
        self,
        num_classes=1,
        n_instance=5,
        center_loss_feat_dim=512,
        base_model=senet_mod(se_resnext50_32x4d, pretrained=True, in_channel=3),
    ):
        super().__init__()
        self.n_instance = n_instance
        self.criterion = nn.CrossEntropyLoss()
        self.center_loss_feat_dim = center_loss_feat_dim
        self.model_name = base_model.__class__.__name__
        self.encoder, nc = drop_fc(base_model)
        self.pool = nn.Sequential(
            AdaptiveConcatPool2d(),
        )
        hidden_dim = 256
        self.lstm = nn.LSTM(nc, hidden_dim, num_layers=2, dropout=0, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2*n_instance, num_classes)

    def forward(self, x):
        bs, n, ch, w, h = x.shape # 8, 5, 3, 384, 384
        x = x.view(bs * n, ch, w, h) # 40, 3, 384, 384
        x = self.encoder(x) # 40, 384
        x = x.view(bs, n, -1) # 8, 5, 384
        x, _ = self.lstm(x)   # (8, 5, hidden_dim*2)
        x = x.reshape(bs, -1)
        x = self.fc(x)
        return x
