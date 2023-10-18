from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AutoModel, AutoConfig, XLNetConfig, XLNetModel, AutoModelForCausalLM

import torch
import torch.nn as nn
from pdb import set_trace as st

class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector

class RoBERTaBaseWithTfidf(nn.Module):
    def __init__(self, model_path, num_class):
        super(RoBERTaBase, self).__init__()
        self.in_features = 768
        self.decoder = RobertaModel.from_pretrained(model_path)
        self.head = AttentionHead(self.in_features,self.in_features)
        self.dropout = nn.Dropout(0.1)
        self.process_num = nn.Sequential(
            nn.Linear(10, 8),
            nn.BatchNorm1d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.process_tfidf = nn.Sequential(
            nn.Linear(100, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.l0 = nn.Linear(self.in_features + 8 + 32, num_class)
        self.l1 = nn.Linear(self.in_features + 8 + 32, 7)

    def forward(self, ids, mask, numerical_features, tfidf):
        roberta_outputs = self.decoder(
            ids,
            attention_mask=mask
        )

        x1 = self.head(roberta_outputs[0]) # bs, 1024

        x2 = self.process_num(numerical_features) # bs, 8

        x3 = self.process_tfidf(tfidf) # bs, 32

        x = torch.cat([x1, x2, x3], 1) # bs, 1024 + 8 + 32

        logits = self.l0(self.dropout(x))
        aux_logits = torch.sigmoid(self.l1(self.dropout(x)))
        return logits.squeeze(-1), aux_logits

class TransformersModel(nn.Module):
    def __init__(self, model_path, num_class, apply_dropout=True):
        super(TransformersModel, self).__init__()
        self.apply_dropout = apply_dropout
        if 'nlp_models/xlnet' not in model_path:
            self.in_features = AutoConfig.from_pretrained(model_path).hidden_size
            print('in_features:', self.in_features)
        if apply_dropout:
            self.decoder = AutoModel.from_pretrained(
                model_path,
                output_hidden_states=True
            )
        else:
            if ('nlp_models/roberta' in model_path) | ('nlp_models/deberta' in model_path) | ('nlp_models/albert_v2' in model_path):
                self.decoder = AutoModel.from_pretrained(model_path, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
            elif 'nlp_models/electra' in model_path:
                self.decoder = AutoModel.from_pretrained(model_path, hidden_dropout_prob=0, attention_probs_dropout_prob=0, summary_last_dropout=0)
            elif 'nlp_models/bart' in model_path:
                self.decoder = AutoModel.from_pretrained(model_path, dropout=0)
            elif 'nlp_models/bart' in model_path:
                self.decoder = AutoModel.from_pretrained(model_path, dropout=0)
            elif 'nlp_models/gpt2' in model_path:
                self.decoder = AutoModel.from_pretrained(model_path,
                                                    attn_pdrop=0,
                                                    embd_pdrop=0,
                                                    resid_pdrop=0,
                                                    summary_first_dropout=0)
            elif 'nlp_models/xlnet' in model_path:
                if 'base' in model_path:
                    xlnet_config = XLNetConfig.from_json_file('nlp_models/xlnet/xlnet-base-cased-config.json')
                else:
                    xlnet_config = XLNetConfig.from_json_file('nlp_models/xlnet/xlnet-large-cased-config.json')
                if not apply_dropout:
                    xlnet_config.hidden_dropout_prob = 0
                    xlnet_config.attention_probs_dropout_prob = 0
                    xlnet_config.dropout = 0
                self.decoder = XLNetModel.from_pretrained(model_path, config=xlnet_config)
                self.in_features = xlnet_config.hidden_size
            else:
                raise

        self.head = AttentionHead(self.in_features, self.in_features)
        self.l0 = nn.Linear(self.in_features, num_class)
        if apply_dropout:
            self.dropout = nn.Dropout(0.1)
            self.ln = nn.LayerNorm(self.in_features)

    def forward(self, ids, mask, token_type_ids):
        x = self.decoder(
            input_ids=ids,
            attention_mask=mask,
            # token_type_ids = token_type_ids
        )
        if self.apply_dropout:
            x = x['last_hidden_state']
            x = torch.mean(x, axis=1)
            # x = self.dropout(self.ln(x))
            x = self.ln(x)

        else:
            x = self.head(x[0]) # bs, in_features
        x = self.l0(x)
        return x


        last_hidden_state, pooler_output, hidden_state = self.decoder(input_ids=ids,
                                 attention_mask=mask,
                                 token_type_ids = token_type_ids,
                                 return_dict=False)
#         cls_token = last_hidden_state[:, 0, :]
        cls_token = torch.mean(last_hidden_state, axis=1)

        # cls_token = torch.cat([
        #            hidden_state[-1].mean(dim=1),
        #            hidden_state[-1][:, 0, :],
        #            hidden_state[-2][:, 0, :],
        #            hidden_state[-3][:, 0, :],
        #            hidden_state[-4][:, 0, :]
        # ], dim=1)
        output = self.l0(cls_token)
        return output


    def forward_with_features(self, ids, mask, token_type_ids):
        x = self.decoder(
            input_ids=ids,
            attention_mask=mask,
            # token_type_ids = token_type_ids
        )
        if self.apply_dropout:
            x = x['last_hidden_state']
            x = torch.mean(x, axis=1)
            features = self.dropout(self.ln(x))

        else:
            features = self.head(x[0]) # bs, in_features
        x = self.l0(features)
        return x, features

class TransformersModelForFalcon7b(nn.Module):
    def __init__(self, model_path, num_classes, dropout=0.3):
        super(TransformersModelForFalcon7b, self).__init__()
        self.in_features = 65024
        self.decoder = AutoModelForCausalLM.from_pretrained("h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3", torch_dtype=torch.float16, device_map={"": "cuda:0"},trust_remote_code=True)
        # self.decoder = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map={"": "cuda:0"},trust_remote_code=True)
        self.ln0 = nn.Linear(self.in_features, 256)
        self.ln1 = nn.Linear(256, num_classes)
        self.ln0.to(torch.float16)
        self.ln1.to(torch.float16)

    def forward(self, ids, mask, token_type_ids):
        x = self.decoder(
            input_ids=ids,
            attention_mask=mask,
        )
        # st()
        x = self.ln0(x.logits)
        x = x.mean(dim=1)
        x = self.ln1(x)
        return x

class TransformersModelMetacols(nn.Module):
    def __init__(self, model_path, num_class, apply_dropout=True, input_num=0):
        super(TransformersModelMetacols, self).__init__()
        self.apply_dropout = apply_dropout
        if 'nlp_models/xlnet' not in model_path:
            self.in_features = AutoConfig.from_pretrained(model_path).hidden_size
            print('in_features:', self.in_features)
        if apply_dropout:
            self.decoder = AutoModel.from_pretrained(model_path)
        else:
            if ('nlp_models/roberta' in model_path) | ('nlp_models/deberta' in model_path) | ('nlp_models/albert_v2' in model_path):
                self.decoder = AutoModel.from_pretrained(model_path, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
            elif 'nlp_models/electra' in model_path:
                self.decoder = AutoModel.from_pretrained(model_path, hidden_dropout_prob=0, attention_probs_dropout_prob=0, summary_last_dropout=0)
            elif 'nlp_models/bart' in model_path:
                self.decoder = AutoModel.from_pretrained(model_path, dropout=0)
            elif 'nlp_models/bart' in model_path:
                self.decoder = AutoModel.from_pretrained(model_path, dropout=0)
            elif 'nlp_models/gpt2' in model_path:
                self.decoder = AutoModel.from_pretrained(model_path,
                                                    attn_pdrop=0,
                                                    embd_pdrop=0,
                                                    resid_pdrop=0,
                                                    summary_first_dropout=0)
            elif 'nlp_models/xlnet' in model_path:
                if 'base' in model_path:
                    xlnet_config = XLNetConfig.from_json_file('nlp_models/xlnet/xlnet-base-cased-config.json')
                else:
                    xlnet_config = XLNetConfig.from_json_file('nlp_models/xlnet/xlnet-large-cased-config.json')
                if not apply_dropout:
                    xlnet_config.hidden_dropout_prob = 0
                    xlnet_config.attention_probs_dropout_prob = 0
                    xlnet_config.dropout = 0
                self.decoder = XLNetModel.from_pretrained(model_path, config=xlnet_config)
                self.in_features = xlnet_config.hidden_size
            else:
                raise
        self.input_num = input_num
        if input_num != 0:
            self.mlp = nn.Sequential(
                nn.BatchNorm1d(input_num),
                nn.Linear(input_num, 8),
                nn.BatchNorm1d(8),
                nn.Dropout(p=0.3),
                nn.ReLU(),
                nn.Linear(8, 16),
                nn.BatchNorm1d(16),
                nn.Dropout(p=0.3),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.BatchNorm1d(8),
                nn.Dropout(p=0.3),
            )

        self.head = AttentionHead(self.in_features, self.in_features)
        if input_num != 0:
            head_output_num = 8
        else:
            head_output_num = 0

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.in_features+head_output_num),
            nn.Linear(self.in_features+head_output_num, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(256, num_class),
        )

        if apply_dropout:
            self.dropout = nn.Dropout(0.1)
            self.ln = nn.LayerNorm(self.in_features)

    def forward(self, ids, mask, numerical_features):
        x = self.decoder(
            ids,
            attention_mask=mask
        )
        if self.input_num != 0:
            x2 = self.mlp(numerical_features) # bs, 128
        if self.apply_dropout:
            x = x['last_hidden_state']
            # x = x[:, 0, :]
            x = torch.mean(x, axis=1)
            x = self.ln(x)
            #x = self.dropout(x)
            x = self.l0(self.dropout(x))
        else:
            if self.input_num != 0:
                x = self.head(x[0]) # bs, in_features
                x = torch.cat([x, x2], 1)
            else:
                x = x[0]
                x = self.head(x) # bs, in_features
            x = self.fc(x)
        return x
