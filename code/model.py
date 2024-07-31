from transformers import BertModel, BertConfig, RobertaModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import GraphAttentionLayer
from torch.autograd import Variable
import numpy as np

class CSTE_for_medical(nn.Module):
    def __init__(self, configs):
        super(CSTE_for_medical, self).__init__()

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.hidden_size = configs.bert_output_dim
        self.bert_0 = BertModel.from_pretrained(configs.roberta_cache_path_Chinese)
        self.bert_1 = BertModel.from_pretrained(configs.roberta_cache_path_Chinese)

        self.fc_bert = nn.Linear(configs.bert_output_dim, configs.lstm_hidden_dim * 2)

        self.dd1 = nn.Linear(configs.lstm_hidden_dim * 2, configs.lstm_hidden_dim * 2, bias=False)
        self.dd2 = nn.Linear(configs.lstm_hidden_dim * 2, configs.lstm_hidden_dim * 2, bias=False)
        self.dd3 = nn.Linear(configs.lstm_hidden_dim * 2, configs.lstm_hidden_dim * 2, bias=False)

        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.lstm_hidden_dim, num_layers=1, batch_first=True,
                                  bidirectional=True)  # 双向LSTM
        self.lstm_layer1 = nn.LSTM(configs.bert_output_dim, configs.lstm_hidden_dim, num_layers=1, batch_first=True,
                                  bidirectional=True)  # 双向LSTM
        self.lstm_layer_context = nn.LSTM(configs.bert_output_dim, configs.lstm_hidden_dim, num_layers=1, batch_first=True,
                                  bidirectional=True)  # 双向LSTM
        self.fc_final1 = torch.nn.Linear(configs.lstm_hidden_dim * 12, 512, bias=False)
        self.fc_final2 = torch.nn.Linear(512, configs.num_classes, bias=False)


    def forward(self, bert_token, bert_mask, token_type, length, bert_token_summary, bert_mask_summary,
                token_type_summary):
        bert_output_summary = self.bert_0(
            input_ids=bert_token_summary.long(),
            attention_mask=bert_mask_summary.long(),
            token_type_ids=token_type_summary.long())#
        patient_pooler_output = bert_output_summary[0] #[batch, length, 768]
        patient_lstm_output, (final_hidden_state1, final_cell_state1) = self.lstm_layer(patient_pooler_output) #[batch,length,512]


        patient_cls = self.fc_bert(bert_output_summary[0][:, 0, :])#[batch, 768]
        d_out1 = patient_cls
        g1 = self.dd1(d_out1)
        d_out1 = g1 * d_out1 + g1
        g2 = self.dd1(d_out1)
        d_out1 = g2 * d_out1 + g2

        patient_cat = torch.cat([d_out1, patient_lstm_output[:, -1, :]],dim=1)

        bert_output_0 = self.bert_1(
            input_ids=(bert_token[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).long(),
            attention_mask=(bert_mask[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).long(),
            token_type_ids=(token_type[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).long())  # not sure
        doctor_pooler_output = bert_output_0[0]

        doctor_lstm_output, (final_hidden_state2, final_cell_state2) = self.lstm_layer1(doctor_pooler_output)

        doctor_cls = self.fc_bert(bert_output_0[0][:, 0, :])
        d_out2 = doctor_cls
        d1 = self.dd2(d_out2)
        d_out2 = d1 * d_out2 + d1
        d2 = self.dd2(d_out2)
        d_out2 = d2 * d_out2 + d2

        doctor_cat = torch.cat([d_out2, doctor_lstm_output[:, -1, :]], dim=1)
        final_cat = torch.cat([doctor_cat, patient_cat], dim=1)

        contextual_cat =  torch.cat([patient_pooler_output,doctor_pooler_output],dim=1) #batch ,len1+len2,768
        contextual_lstm_output, (final_hidden_state3, final_cell_state3) =self.lstm_layer_context(contextual_cat)# batch, len1+len2, 512*2
        contextual_lstm_output_mean = torch.mean(contextual_lstm_output, dim=1)# b, 1024

        context_change = self.fc_bert(contextual_cat) #b,lenall,1024
        d_out3 = context_change
        c1 = self.dd3(d_out3)
        d_out3 = c1 * d_out3 + c1
        c2 = self.dd3(d_out3)
        d_out3 = c2 * d_out3 + c2
        context_change_mean = torch.mean(d_out3, dim=1)
        contextual_final = torch.cat([context_change_mean,contextual_lstm_output_mean], dim=1)


        final_cat_add_context = torch.cat([contextual_final,final_cat], dim=1)
        transformer_output = F.leaky_relu(self.fc_final2(self.fc_final1(final_cat_add_context)))
        return transformer_output


    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss

