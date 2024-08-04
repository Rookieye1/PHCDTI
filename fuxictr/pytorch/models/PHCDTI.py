


# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from torch import nn
import numpy as np
import torch
from PHCDTI.FuxiCTR.fuxictr.pytorch.models import BaseModel
from PHCDTI.FuxiCTR.fuxictr.pytorch.layers import MLP_Layer, EmbeddingLayer, MultiHeadSelfAttention, SqueezeExcitationLayer, PairInteractionLayer, LR_Layer,CompressedInteractionNet


class PHCDTI(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="PHCDTI",
                 gpu=1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[1024, 512, 512],
                 reduction_ratio=0.7,
                 bilinear_type="field_interaction",
                 dnn_activations="ReLU",
                 attention_layers=10,
                 num_heads=8,
                 attention_dim=20,
                 net_dropout=0,
                 conv=64,
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=True,
                 batch_size = 32,
                 protein_MAX_LENGH=1000,
                 drug_MAX_LENGH=100,
                 protein_kernel=[4, 8, 12],
                 drug_kernel=[4, 6, 8],
                 use_residual=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(PHCDTI, self).__init__(feature_map,
                                      model_id=model_id,
                                      gpu=gpu,
                                      embedding_regularizer=embedding_regularizer,
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        self.drug_kernel = drug_kernel
        self.protein_kernel= protein_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.conv = conv
        self.bz = batch_size
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.num_fields = feature_map.num_fields
        self.senet_layer = SqueezeExcitationLayer(self.num_fields, reduction_ratio)
        self.bilinear_interaction = BilinearInteractionLayer(self.num_fields, embedding_dim, bilinear_type)
        self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=False)
        self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=False)
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.embedding_dim*2, kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.embedding_dim*2, out_channels=self.embedding_dim * 2, kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.embedding_dim * 2, out_channels=self.embedding_dim, kernel_size=self.drug_kernel[2]),
            nn.ReLU())
        self.Drug_max_pool = nn.MaxPool1d(self.drug_MAX_LENGH - self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.conv*2, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv, kernel_size=self.protein_kernel[2]),
            nn.ReLU())
        self.Protein_max_pool = nn.MaxPool1d(self.protein_MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)
        self.dnn = MLP_Layer(input_dim= (self.num_fields * (self.num_fields - 1) )* embedding_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm,
                             use_bias=True) \
                   if dnn_hidden_units else None # in case no DNN used
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(embedding_dim if i == 0 else num_heads * attention_dim,
                                    attention_dim=attention_dim,
                                    num_heads=num_heads,
                                    dropout_rate=net_dropout,
                                    use_residual=use_residual,
                                    use_scale=use_scale,
                                    layer_norm=layer_norm,
                                    align_to="output")
             for i in range(attention_layers)])
        self.fc = nn.Linear(feature_map.num_fields * attention_dim * num_heads, 1)
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        # print(y[0][0] == 0)
        # print(feature_emb.shape)
        drug = X[:, 0].long()
        protein = X[:, 1].long()
        drug_embedding_layer = nn.Embedding(int(max(X[:, 0])) + 1, self.embedding_dim)
        protein_embedding_layer = nn.Embedding(int(max(X[:, 1])) + 1, self.embedding_dim)
        drugembed = drug_embedding_layer(drug).unsqueeze(1)
        proteinembed = protein_embedding_layer(protein).unsqueeze(1)
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)
        feature_emb = self.embedding_layer(X)
        senet_emb = self.senet_layer(feature_emb)  # size = [bz,nf,emb_dim]
        bilinear_p = self.bilinear_interaction(feature_emb)  # size = [bz,C^(2)_(nf),emb_dim]
        bilinear_q = self.bilinear_interaction(senet_emb)  # size = [bz,C^(2)_(nf),emb_dim]
        attention_out = self.self_attention(feature_emb) #size=[bz,nf,num_head*att_dim]
        # print(attention_out.shape)
        attention_out = torch.flatten(attention_out, start_dim=1)#size=[bz,nf*num_head*att_dim]
        # fig_attention_out = torch.mean(attention_out.reshape(self.bz, self.num_heads*self.num_fields, self.attention_dim), axis=1)
        pair_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)# size = [bz,nf*(nf-1)*emb_dim]
        # fig_bilinear_out = torch.mean(pair_out.reshape(self.bz, self.num_fields*(self.num_fields-1), self.embedding_dim), axis=1)
        dnn_out = self.dnn(pair_out)
        att_out = self.fc(attention_out)
        y_pred = self.output_activation(dnn_out + att_out + self.lr_layer(X))
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict



# import torch
# from torch import nn
# from FuxiCTR_Project.FuxiCTR.fuxictr.pytorch.models import BaseModel
# from FuxiCTR_Project.FuxiCTR.fuxictr.pytorch.layers import EmbeddingLayer, InnerProductLayer, LR_Layer,MultiHeadSelfAttention
# from FuxiCTR_Project.FuxiCTR.fuxictr.pytorch.layers import MLP_Layer
# import torch.nn.functional as F
#
# class HOA(BaseModel):
#     def __init__(self,
#                  feature_map,
#                  model_id="HOA",
#                  gpu=-1,
#                  task="binary_classification",
#                  learning_rate=1e-3,
#                  embedding_dim=10,
#                  num_filters=128,
#                  protein_MAX_LENGH=1000,
#                  drug_MAX_LENGH=100,
#                  protein_kernel=[4, 8, 12],
#                  filter_size = 3,
#                  conv= 64,
#                  dnn_hidden_units=[64, 64, 64],
#                  drug_kernel=[4, 6, 8],
#                  dnn_activations="ReLU",
#                  attention_layers=10,
#                  num_heads=8,
#                  attention_dim=20,
#                  net_dropout=0,
#                  batch_norm=False,
#                  layer_norm=False,
#                  use_scale=False,
#                  use_wide=False,
#                  use_residual=True,
#                  embedding_regularizer=None,
#                  net_regularizer=None,
#                  item_tower_dropout=0,
#                  **kwargs):
#         super(HOA, self).__init__(feature_map,
#                                    model_id=model_id,
#                                    gpu=gpu,
#                                    embedding_regularizer=embedding_regularizer,
#                                    net_regularizer=net_regularizer,
#                                    **kwargs)
#         self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
#         self.embedding_dim = embedding_dim
#         self.num_filters = num_filters
#         self.filter_size = filter_size
#         self.num_field = feature_map.num_fields
#         self.protein_MAX_LENGH = protein_MAX_LENGH
#         self.drug_MAX_LENGH = drug_MAX_LENGH
#         self.self_attention = nn.Sequential(
#             *[MultiHeadSelfAttention(embedding_dim if i == 0 else num_heads * attention_dim,
#                                     attention_dim=attention_dim,
#                                     num_heads=num_heads,
#                                     dropout_rate=net_dropout,
#                                     use_residual=use_residual,
#                                     use_scale=use_scale,
#                                     layer_norm=layer_norm,
#                                     align_to="output")
#              for i in range(attention_layers)])
#         self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=False) \
#                         if use_wide else None
#         self.dnn = MLP_Layer(input_dim=embedding_dim * feature_map.num_fields,
#                              output_dim=1,
#                              hidden_units=dnn_hidden_units,
#                              hidden_activations=dnn_activations,
#                              output_activation=None,
#                              dropout_rates=net_dropout,
#                              batch_norm=batch_norm,
#                              use_bias=True) \
#                    if dnn_hidden_units else None # in case no DNN used
#         self.conv1d_layer = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_filters,
#                                  kernel_size=self.filter_size)
#         self.embedding_dim = embedding_dim
#         self.conv = conv
#         self.drug_MAX_LENGH = drug_MAX_LENGH
#         self.drug_kernel = drug_kernel
#         self.protein_MAX_LENGH = protein_MAX_LENGH
#         self.protein_kernel = protein_kernel
#         self.num_fields = feature_map.num_fields
#
#         self.protein_embed = nn.Embedding(26, self.embedding_dim, padding_idx=0)
#         self.drug_embed = nn.Embedding(65, self.embedding_dim, padding_idx=0)
#         self.Drug_CNNs = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=self.embedding_dim*2, kernel_size=self.drug_kernel[0]),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=self.embedding_dim*2, out_channels=self.embedding_dim * 2, kernel_size=self.drug_kernel[1]),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=self.embedding_dim * 2, out_channels=self.embedding_dim, kernel_size=self.drug_kernel[2]),
#             nn.ReLU(),
#         )
#         self.Drug_max_pool = nn.MaxPool1d(
#             self.drug_MAX_LENGH - self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3)
#         self.Protein_CNNs = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=self.conv*2, kernel_size=self.protein_kernel[0]),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=self.conv*2, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv, kernel_size=self.protein_kernel[2]),
#             nn.ReLU(),
#         )
#         self.Protein_max_pool = nn.MaxPool1d(
#             self.protein_MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)
#
#         self.fc = nn.Linear(feature_map.num_fields * attention_dim * num_heads, 1)
#         self.output_activation = self.get_output_activation(task)
#         self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
#         self.reset_parameters()
#         self.model_to_device()
#
#     def forward(self, inputs):
#         X, y = self.inputs_to_device(inputs)
#         drug = X[:, 0].long()
#         protein = X[:, 1].long()
#         drug_embedding_layer = nn.Embedding(int(max(X[:, 0])) + 1, self.embedding_dim)
#         protein_embedding_layer = nn.Embedding(int(max(X[:, 1])) + 1, self.embedding_dim)
#         # print(drug.shape)
#         drugembed = drug_embedding_layer(drug).unsqueeze(1)
#         proteinembed = protein_embedding_layer(protein).unsqueeze(1)
#         # drugembed = drugembed.permute(0, 2, 1)
#         # proteinembed = proteinembed.permute(0, 2, 1)
#         # print(proteinembed.shape)
#         drugConv = self.Drug_CNNs(drugembed)
#         proteinConv = self.Protein_CNNs(proteinembed)
#         # print(drugConv.shape)
#         # print(proteinConv.shape)
#         # drugConv = drugConv.squeeze(2).permute(0, 2, 1)
#         # proteinConv= proteinConv.squeeze(2).permute(0, 2, 1)
#         proteinConv = F.adaptive_avg_pool1d(proteinConv, output_size=int(self.num_fields / 2))
#         drugConv = F.adaptive_avg_pool1d(drugConv, output_size=int(self.num_fields / 2))
#         # print(proteinConv.shape)
#         # print(drugConv.shape)
#         feature_emb = torch.cat((drugConv, proteinConv), dim=-1)
#         feature_emb1 = feature_emb.permute(0, 2, 1)
#         attention_out = self.self_attention(feature_emb1)
#         # print(attention_out.shape)
#         attention_out = torch.flatten(attention_out, start_dim=1)
#         # print(attention_out.shape)
#         y_pred = self.fc(attention_out)
#         if self.dnn is not None:
#             y_pred += self.dnn(feature_emb.flatten(start_dim=1))
#         if self.lr_layer is not None:
#             y_pred += self.lr_layer(X)
#         if self.output_activation is not None:
#             y_pred = self.output_activation(y_pred)
#         return_dict = {"y_true": y, "y_pred": y_pred}
#         return return_dict
#
#         # X, y = self.inputs_to_device(inputs)
#         # drug = X[:,0].long()
#         # protein = X[:,1].long()
#         # drug_embedding_layer = nn.Embedding(int(max(X[:, 0])) + 1, self.embedding_dim)
#         # protein_embedding_layer = nn.Embedding(int(max(X[:, 1])) + 1, self.embedding_dim)
#         # # print(drug.shape)
#         # drugembed = drug_embedding_layer(drug).unsqueeze(1).expand(-1,self.drug_MAX_LENGH , -1)
#         # proteinembed = protein_embedding_layer(protein).unsqueeze(1).expand(-1, self.protein_MAX_LENGH, -1)
#         # drugembed = drugembed.permute(0, 2, 1)
#         # proteinembed = proteinembed.permute (0, 2, 1)
#         # # print(proteinembed.shape)
#         # drugConv = self.Drug_CNNs(drugembed)
#         # proteinConv = self.Protein_CNNs(proteinembed)
#         # proteinConv = F.adaptive_avg_pool1d(proteinConv, output_size=int(self.num_fields/2))
#         # drugConv = F.adaptive_avg_pool1d(drugConv, output_size=int(self.num_fields/2))
#         # # print(proteinConv.shape)
#         # feature_emb = torch.cat((drugConv, proteinConv), dim=-1)
#         # feature_emb1 = feature_emb.permute(0, 2, 1)
#         # attention_out = self.self_attention(feature_emb1)
#         # # print(attention_out.shape)
#         # attention_out = torch.flatten(attention_out, start_dim=1)
#         # # print(attention_out.shape)
#         # y_pred = self.fc(attention_out)
#         # if self.dnn is not None:
#         #     y_pred += self.dnn(feature_emb.flatten(start_dim=1))
#         # if self.lr_layer is not None:
#         #     y_pred += self.lr_layer(X)
#         # if self.output_activation is not None:
#         #     y_pred = self.output_activation(y_pred)
#         # return_dict = {"y_true": y, "y_pred": y_pred}
#         # return return_dict
