import numpy as np
from random import sample
import torch.nn.init as init
from torch.autograd import Variable
import time
from numbers import Number
import math
import logging
from typing import Optional, Tuple
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torch.nn import L1Loss, MSELoss
from torch.autograd import Function
from math import pi, log
from functools import wraps
import os
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.activations import gelu, gelu_new
from transformers import BertConfig
import numpy as np
import torch.optim as optim
from itertools import chain
from model_utils import *
from utils import *
import torch
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from math import pi, log
from functools import wraps
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from modules.transformer import TransformerEncoder




logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
]


ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
}
   
class BERT_TM(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.config.output_hidden_states=True
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # fused_embedding = embedding_output

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_sequence_output = encoder_outputs[0]# 36*60*768:bsz*msl*dim
        # print('last_sequence_output_shape', last_sequence_output.shape)# [256,50,768]
        return last_sequence_output




class HME(BertPreTrainedModel):
    def __init__(self, config, args = None):
        super().__init__(config)
        self.bert = BERT_TM(config, args)
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.TEXT_DIM, args.ACOUSTIC_DIM, args.VISUAL_DIM
        self.d_l = self.d_a = self.d_v = args.d_l
        self.args = args
        self.bsz = args.train_batch_size 
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.latent_depth = args.latent_layers
        self.activation = nn.ReLU()
        self.missing_rate = args.missing_rate
        self.similarity_threshold = args.similarity_threshold
################################################################################################
        # =========================
# Top-k similarity enhancement parameters
# =========================
# top_k:
# 每个样本从当前 batch 里找最相似的 top_k 个样本来增强。
# 这里用 getattr 是为了防止 HME_main_topk.py 还没加参数时报错。
        self.top_k = getattr(args, 'top_k', 3)
# top_temperature:
# softmax 加权时的温度参数。
# 值越小，越偏向最相似的样本；
# 值越大，几个 top-k 样本的权重越平均。
        self.top_temperature = getattr(args, 'top_temperature', 0.1)
        # Whether to use quality scores in top-k enhancement
        self.use_quality_weight = getattr(args, 'use_quality_weight', True)

        self.num_latents = args.num_latents
        self.hyper_depth = args.hyper_depth
        self.output_dim = args.output_dim
        self.LB = args.uncertainty_LB
        self.UB = args.uncertainty_UB

        # within-modality interactions
        self.l2l = self.Get_Modality_Network(modality='ll')
        self.a2a = self.Get_Modality_Network(modality='aa')
        self.v2v = self.Get_Modality_Network(modality='vv')
        
        # cross-modality interactions
        self.l2a = self.get_hyper_modality(self_type='l2a')
        self.l2v = self.get_hyper_modality(self_type='l2v')
        self.a2l = self.get_hyper_modality(self_type='a2l')
        self.a2v = self.get_hyper_modality(self_type='a2v')
        self.v2l = self.get_hyper_modality(self_type='v2l')
        self.v2a = self.get_hyper_modality(self_type='v2a')

        self.hyper_l = self.get_hyper_modality(self_type='hyper_l')
        self.hyper_a = self.get_hyper_modality(self_type='hyper_a')
        self.hyper_v = self.get_hyper_modality(self_type='hyper_v')

        self.latent_sample = self.get_latent_network(self_type='sam')
        self.latent_text = self.get_latent_network(self_type='mod_l')
        self.latent_audio = self.get_latent_network(self_type='mod_a')
        self.latent_video = self.get_latent_network(self_type='mod_v')

        # 1. Temporal convolutional layers
        self.proj_l_1 = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)#
        self.proj_a_1 = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)#
        self.proj_v_1 = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)#

        # VIB estimation
        self.VIB_estimate_l_p = VIB_estimate(self.d_l, self.d_l, self.output_dim)
        self.VIB_estimate_a_p = VIB_estimate(self.d_a, self.d_a, self.output_dim)
        self.VIB_estimate_v_p = VIB_estimate(self.d_v, self.d_v, self.output_dim)
        
        self.pool_audio = nn.AdaptiveMaxPool1d(1)
        self.pool_text = nn.AdaptiveMaxPool1d(1)
        self.pool_video = nn.AdaptiveMaxPool1d(1)
        self.dropout = args.dropout
        
        # Integrate the hyper into modality
        self.ATTN_T = Attention_Uncertainty_2(self.d_l, self.d_l, self.LB, self.UB)
        self.ATTN_A = Attention_Uncertainty_2(self.d_a, self.d_a, self.LB, self.UB)
        self.ATTN_V = Attention_Uncertainty_2(self.d_v, self.d_v, self.LB, self.UB)

        self.ATTN_hyper = Attention_Uncertainty_3(self.d_l, self.d_l)
        
        encoder_layer_all = nn.TransformerEncoderLayer(d_model=self.d_l, nhead=self.num_heads)
        self.transformer_encoder_all = nn.TransformerEncoder(encoder_layer_all, num_layers=2)

        self.fusion_all = nn.Sequential()
        self.fusion_all.add_module('fusion_layer_all', nn.Linear(in_features=self.d_l*4, out_features=self.d_l*2))
        self.fusion_all.add_module('fusion_layer_all_dropout', nn.Dropout(self.dropout))
        self.fusion_all.add_module('fusion_layer_all_activation', self.activation)
        self.fusion_all.add_module('fusion_layer_all_all', nn.Linear(in_features=self.d_l*2, out_features=self.output_dim))  
        self.init_weights()
        
    def Get_Modality_Network(self, modality='l'):
        if modality in ['ll', 'al', 'vl']:
            attn_dropout = self.attn_dropout
        elif modality in ['aa', 'la', 'va']:
            attn_dropout = self.attn_dropout_a
        elif modality in ['vv', 'lv', 'av']:
            attn_dropout = self.attn_dropout_v
        else:
            raise ValueError("Unknown network type")
            
        return TransformerEncoder(embed_dim=self.d_l,
                                  num_heads=self.num_heads,
                                  layers=self.layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=False)

    def get_latent_network(self, self_type='sam',layers=-1):
        assert self_type in ['sam', 'mod_l', 'mod_a', 'mod_v']
        
        return LatentTransformer(num_latents=self.num_latents,
                                 latent_dim=self.d_l,
                                 input_dim=self.d_l,
                                 depth=self.latent_depth,
                                 heads=self.num_heads,
                                 latent_heads=self.num_heads,
                                 latent_dim_head=self.d_l//self.num_heads,
                                 dim_head=self.d_l//self.num_heads,
                                 )

    def get_hyper_modality(self, self_type='sam',layers=-1):
        assert self_type in ['l2a', 'l2v', 'a2l', 'a2v', 'v2l', 'v2a', 'hyper_l', 'hyper_a', 'hyper_v']
        # input_dim, depth, heads, dim_head, latent_heads, latent_dim_head
        return Encoder_Layer(input_dim=self.d_l,
                             depth=self.hyper_depth,
                             heads=self.num_heads,
                             dim_head=self.d_l//self.num_heads,
                             latent_heads=self.num_heads,
                             latent_dim_head=self.d_l//self.num_heads,
        )
    def apply_modality_degradation(self, x, degradation_level):
        """
        Apply multi-level degradation to one modality.

        x:
            模态特征，形状一般是 [B, T, D]
        degradation_level:
            当前模态的退化等级，形状是 [B]
            0 = 完整
            1 = 轻度退化
            2 = 中度退化
            3 = 重度退化
            4 = 完全缺失

        这里先做第一版：
            轻度退化：少量特征 dropout + 小噪声
            中度退化：中等特征 dropout + 中等噪声
            重度退化：大量特征 dropout + 较大噪声
            完全缺失：直接置零
        """

        if degradation_level.dim() > 1:
            degradation_level = degradation_level.view(-1)

        degradation_level = degradation_level.long()
        x_degraded = x.clone()

        # 不同退化等级对应的 dropout 概率和噪声强度
        degradation_config = {
            1: {"dropout": 0.10, "noise": 0.05},
            2: {"dropout": 0.30, "noise": 0.10},
            3: {"dropout": 0.50, "noise": 0.20},
        }

        # 1/2/3：轻度、中度、重度退化
        for level, cfg in degradation_config.items():
            sample_mask = degradation_level == level

            if sample_mask.any():
                cur_x = x_degraded[sample_mask]

                # feature dropout：随机丢掉一部分特征
                keep_mask = (torch.rand_like(cur_x) > cfg["dropout"]).float()

                # gaussian noise：加一点噪声，模拟低质量特征
                noise = torch.randn_like(cur_x) * cfg["noise"]

                x_degraded[sample_mask] = cur_x * keep_mask + noise

        # 4：完全缺失，直接置零
        missing_mask = degradation_level >= 4
        if missing_mask.any():
            x_degraded[missing_mask] = 0.0

        return x_degraded

    def generate_degraded_modalities(self, degradation_matrix, texts, acoustic, visual):
        """
        Generate degraded modalities according to degradation_matrix.

        degradation_matrix:
            形状是 [B, 3]
            第 0 列：文本模态退化等级
            第 1 列：音频模态退化等级
            第 2 列：视觉模态退化等级

        输出：
            x_l, x_a, x_v
        """

        degradation_matrix = degradation_matrix.long()

        text_level = degradation_matrix[:, 0]
        acoustic_level = degradation_matrix[:, 1]
        visual_level = degradation_matrix[:, 2]

        x_l = self.apply_modality_degradation(texts, text_level)
        x_a = self.apply_modality_degradation(acoustic, acoustic_level)
        x_v = self.apply_modality_degradation(visual, visual_level)

        return x_l, x_a, x_v
    def degradation_to_quality(self, degradation_level):
        """
        Convert degradation level to quality score.

        degradation_level:
            0 = 完整
            1 = 轻度退化
            2 = 中度退化
            3 = 重度退化
            4 = 完全缺失

        quality score:
            质量越高，分数越大；
            完全缺失的质量分数为 0。
        """

        if degradation_level.dim() > 1:
            degradation_level = degradation_level.view(-1)

        degradation_level = degradation_level.long().clamp(min=0, max=4)
        device = degradation_level.device

        quality_table = torch.tensor(
            [1.0, 0.8, 0.6, 0.4, 0.0],
            device=device,
            dtype=torch.float
        )

        quality_scores = quality_table[degradation_level]

        return quality_scores

    def get_modality_quality_scores(self, degradation_matrix):
        """
        Get quality scores for text, audio and visual modalities.

        degradation_matrix:
            [B, 3]
            第 0 列：文本退化等级
            第 1 列：音频退化等级
            第 2 列：视觉退化等级
        """

        degradation_matrix = degradation_matrix.long()

        quality_l = self.degradation_to_quality(degradation_matrix[:, 0])
        quality_a = self.degradation_to_quality(degradation_matrix[:, 1])
        quality_v = self.degradation_to_quality(degradation_matrix[:, 2])

        return quality_l, quality_a, quality_v
########################################################################################################################   
    def degradation_to_uncertainty_target(self, degradation_level):
        """
        Convert degradation level to uncertainty target.

        0 = 完整       -> 目标不确定性低
        1 = 轻度退化   -> 稍微不确定
        2 = 中度退化   -> 中等不确定
        3 = 重度退化   -> 较高不确定
        4 = 完全缺失   -> 最高不确定

        返回值范围在 0~1 之间。
        """

        if degradation_level.dim() > 1:
            degradation_level = degradation_level.view(-1)

        degradation_level = degradation_level.long().clamp(min=0, max=4)
        device = degradation_level.device

        uncertainty_table = torch.tensor(
            [0.1, 0.3, 0.5, 0.7, 0.9],
            device=device,
            dtype=torch.float
        )

        uncertainty_targets = uncertainty_table[degradation_level]

        return uncertainty_targets

    def reduce_std_to_uncertainty_score(self, std):
        """
        Convert VIB std to sample-level uncertainty score.

        std 可能是 [B, D] 或 [B, 1, D]。
        这里把它压成 [B]，再用 sigmoid 映射到 0~1。
        """

        while std.dim() > 1:
            std = std.mean(dim=-1)

        uncertainty_score = torch.sigmoid(std.float())

        return uncertainty_score

    def compute_degradation_uncertainty_loss(self, std_l, std_a, std_v, degradation_matrix):
        """
        Degradation-aware uncertainty consistency loss.

        目标：
            退化等级越高，目标不确定性越高；
            模型预测出来的 std 也应该越高。
        """

        degradation_matrix = degradation_matrix.long()

        target_l = self.degradation_to_uncertainty_target(degradation_matrix[:, 0])
        target_a = self.degradation_to_uncertainty_target(degradation_matrix[:, 1])
        target_v = self.degradation_to_uncertainty_target(degradation_matrix[:, 2])

        pred_l = self.reduce_std_to_uncertainty_score(std_l)
        pred_a = self.reduce_std_to_uncertainty_score(std_a)
        pred_v = self.reduce_std_to_uncertainty_score(std_v)

        loss_l = F.mse_loss(pred_l, target_l)
        loss_a = F.mse_loss(pred_a, target_a)
        loss_v = F.mse_loss(pred_v, target_v)

        uncertainty_loss = (loss_l + loss_a + loss_v) / 3.0

        return uncertainty_loss   





    def get_similar_embeddings(self, embeddings, alpha=0.8, quality_scores=None, top_k=None, temperature=None):
        """
        Top-k similarity enhancement.

        原始版本：
            找所有相似度 >= alpha 的样本，然后直接平均。

        当前版本：
            1. 计算当前 batch 内样本两两之间的余弦相似度；
            2. 每个样本只选最相似的 top_k 个样本；
            3. 用 softmax 根据相似度分配权重；
            4. 越相似的样本权重越大；
            5. 如果没有找到可靠相似样本，就用 batch 内非零样本均值兜底。
        """

        # 防止 batch_size=1 时 squeeze 把 [1, d] 压成 [d]
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        batch_size, emb_dim = embeddings.size()
        device = embeddings.device
                # =========================
        # Prepare quality scores
        # =========================
        # quality_scores: [B]
        # 质量越高，越适合作为增强来源
        if quality_scores is None:
            quality_scores = torch.ones(batch_size, device=device, dtype=embeddings.dtype)
        else:
            quality_scores = quality_scores.to(device).float().view(-1)
            quality_scores = torch.clamp(quality_scores, min=0.0, max=1.0)

        # 如果当前 batch 只有 1 个样本，就没法找其他相似样本
        if batch_size <= 1:
            return embeddings

        # 如果外面没有传 top_k / temperature，就使用 __init__ 里保存的默认值
        if top_k is None:
            top_k = self.top_k
        if temperature is None:
            temperature = self.top_temperature

        # 保证 top_k 合法：
        # 至少为 1，最多不能超过 batch_size - 1
        top_k = min(max(int(top_k), 1), batch_size - 1)

        # 防止 temperature 为 0，避免除零错误
        temperature = max(float(temperature), 1e-6)

        # =========================
        # Step 1: 计算余弦相似度
        # =========================
        # embeddings: [B, d]
        # norm_embeddings: [B, d]
        # cosine_sim: [B, B]
        norm_embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-8)
        cosine_sim = torch.matmul(norm_embeddings, norm_embeddings.transpose(0, 1))

        # 去掉自己和自己的相似度
        # 因为每个样本不能拿自己来增强自己
        self_mask = torch.eye(batch_size, device=device).bool()
        cosine_sim = cosine_sim.masked_fill(self_mask, float('-inf'))

        # =========================
        # Step 2: 准备兜底向量
        # =========================
        # 有些样本可能是缺失模态，被置成了全 0
        # 所以这里先找 batch 里非零的样本
        non_zero_mask = embeddings.abs().sum(dim=1) > 0

        if non_zero_mask.any():
            fallback_embedding = embeddings[non_zero_mask].mean(dim=0)
        else:
            fallback_embedding = torch.zeros(emb_dim, device=device, dtype=embeddings.dtype)

        # =========================
        # Step 3: 取 top-k 相似样本
        # =========================
        # topk_values: 每个样本对应的 top-k 相似度，形状 [B, top_k]
        # topk_indices: 每个样本对应的 top-k 样本下标，形状 [B, top_k]
        topk_values, topk_indices = torch.topk(cosine_sim, k=top_k, dim=1)

        enhanced_embeddings = torch.zeros_like(embeddings)

        # =========================
        # Step 4: 对每个样本做加权增强
        # =========================
        for i in range(batch_size):
                        # 当前 top-k 候选样本的质量分数
            candidate_quality = quality_scores[topk_indices[i]]

            # 只保留：
            # 1. 相似度超过阈值 alpha
            # 2. 质量分数大于 0 的样本
            valid_mask = (topk_values[i] >= alpha) & (candidate_quality > 0)

            if valid_mask.any():
                selected_indices = topk_indices[i][valid_mask]
                selected_sims = topk_values[i][valid_mask]
                selected_quality = quality_scores[selected_indices]
                selected_embeddings = embeddings[selected_indices]

                # =========================
                # Quality-similarity joint score
                # =========================
                # 原来只看 selected_sims
                # 现在同时看 selected_sims 和 selected_quality
                #
                # 相似度高、质量高：权重大
                # 相似度高、质量低：权重会被压低
                if self.use_quality_weight:
                    # Full DARE-HME:
                    # 使用质量分数，联合考虑“相似度”和“样本质量”
                    # 相似度高、质量高：权重大
                    # 相似度高、质量低：权重会被压低
                    joint_scores = selected_sims * selected_quality
                else:
                    # Ablation:
                    # 不使用质量分数，只看相似度
                    # 用于消融实验：DARE-HME w/o Quality Weight
                    joint_scores = selected_sims

                weights = F.softmax(joint_scores / temperature, dim=0).unsqueeze(1)

                enhanced_embeddings[i] = (weights * selected_embeddings).sum(dim=0)
            else:
                # 如果没有任何可靠相似样本，就用 batch 内非零均值兜底
                enhanced_embeddings[i] = fallback_embedding

        # 对全零样本再兜底一次，避免缺失模态直接传入无效增强
        zero_mask = ~non_zero_mask
        if zero_mask.any():
            enhanced_embeddings[zero_mask] = fallback_embedding
        return enhanced_embeddings

    
    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        ps,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,):
        
        texts = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        x_l, x_a, x_v = self.generate_degraded_modalities(ps, texts, acoustic, visual)
        quality_l, quality_a, quality_v = self.get_modality_quality_scores(ps)
        x_l = x_l.transpose(1, 2)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        
        # Note that the input is incomplete
        proj_x_l = self.proj_l_1(x_l)
        proj_x_a = self.proj_a_1(x_a)
        proj_x_v = self.proj_v_1(x_v)
        
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        # interactions between ori_representations
        trans_text = self.l2l(proj_x_l,proj_x_l,proj_x_l)
        trans_text = trans_text.permute(1,2,0)
        trans_audio = self.a2a(proj_x_a,proj_x_a,proj_x_a)
        trans_audio = trans_audio.permute(1,2,0)
        trans_video = self.v2v(proj_x_v,proj_x_v,proj_x_v)
        trans_video = trans_video.permute(1,2,0)
        
        last_x_l = self.pool_text(trans_text).squeeze()
        last_x_a = self.pool_audio(trans_audio).squeeze()
        last_x_v = self.pool_video(trans_video).squeeze()

        # enhanced representations
        similar_texts = self.get_similar_embeddings(
            last_x_l,
            self.similarity_threshold,
            quality_scores=quality_l
        )
        similar_audios = self.get_similar_embeddings(
            last_x_a,
            self.similarity_threshold,
            quality_scores=quality_a
        )
        similar_videos = self.get_similar_embeddings(
            last_x_v,
            self.similarity_threshold,
            quality_scores=quality_v
        )


        modality_all = torch.stack([last_x_l, last_x_a, last_x_v], dim=1)
        
        eh_l = self.latent_text(similar_texts)# 
        eh_a = self.latent_audio(similar_audios)# 
        eh_v = self.latent_video(similar_videos)# 
        enhance_samples = self.latent_sample(modality_all)
        
        # FUSION OF HYPERS
        hyper_l2a = self.l2a(eh_l.unsqueeze(dim=1), eh_a.unsqueeze(dim=1))
        hyper_l2v = self.l2v(eh_l.unsqueeze(dim=1), eh_v.unsqueeze(dim=1))
        hyper_a2l = self.a2l(eh_a.unsqueeze(dim=1), eh_l.unsqueeze(dim=1))
        hyper_a2v = self.a2v(eh_a.unsqueeze(dim=1), eh_v.unsqueeze(dim=1))
        hyper_v2l = self.v2l(eh_v.unsqueeze(dim=1), eh_l.unsqueeze(dim=1))
        hyper_v2a = self.v2a(eh_v.unsqueeze(dim=1), eh_a.unsqueeze(dim=1))

        # Fusions 
        hyper_ls = self.hyper_l(enhance_samples.unsqueeze(dim=1), torch.stack((hyper_l2a, hyper_l2v), dim=1))
        hyper_as = self.hyper_a(enhance_samples.unsqueeze(dim=1), torch.stack((hyper_a2l, hyper_a2v), dim=1))
        hyper_vs = self.hyper_v(enhance_samples.unsqueeze(dim=1), torch.stack((hyper_v2l, hyper_v2a), dim=1))
        
        # Use the VIB to maintain the essential information
        (mu_l_p, std_l_p), encoding_l_p, logits_l_p, info_loss_l_p = self.VIB_estimate_l_p(hyper_ls)
        (mu_a_p, std_a_p), encoding_a_p, logits_a_p, info_loss_a_p = self.VIB_estimate_a_p(hyper_as)
        (mu_v_p, std_v_p), encoding_v_p, logits_v_p, info_loss_v_p = self.VIB_estimate_v_p(hyper_vs)


        degradation_uncertainty_loss = self.compute_degradation_uncertainty_loss(
            std_l_p,
            std_a_p,
            std_v_p,
            ps
        )
        
        # Fuse the hyper-modalities
        hyper_all = self.ATTN_hyper(encoding_l_p, encoding_a_p, encoding_v_p, std_l_p, std_a_p, std_v_p)
        
        # Fuse the hyper-modalities into ori_representations
        refine_ls = self.ATTN_T(last_x_l, encoding_l_p, std_l_p, self.LB, self.UB)
        refine_as = self.ATTN_A(last_x_a, encoding_a_p, std_a_p, self.LB, self.UB)
        refine_vs = self.ATTN_V(last_x_v, encoding_v_p, std_v_p, self.LB, self.UB)

        all_embed = torch.stack((refine_ls, refine_as, refine_vs, hyper_all), dim=0)
        hidden = self.transformer_encoder_all(all_embed)
        hidden = torch.cat((hidden[0], hidden[1], hidden[2], hidden[3]), dim=1)
        output = self.fusion_all(hidden)
        info_losses = (info_loss_l_p + info_loss_a_p + info_loss_v_p) / 3.0
        res = {
                "logits_l_p": logits_l_p,
                "logits_a_p": logits_a_p,
                "logits_v_p": logits_v_p,
                "predictions": output,
                "info_losses": info_losses,
                "degradation_uncertainty_loss": degradation_uncertainty_loss,
        }

        
        return res
        
    def test(
        self,
        input_ids,
        visual,
        acoustic,
        ps,
        label_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,):

        # =========================
        # Step 1: 文本编码
        # =========================
        # input_ids: 文本 token 编号，形状大致是 [B, T]
        # attention_mask: 标记哪些位置是真实 token，哪些是 padding
        # token_type_ids: BERT 的句子段标记，这里一般全是 0
        # self.bert 会把文本 token 转成 BERT 表示
        # 输出 texts: [B, T, 768]
        # B = batch size, T = 序列长度，一般是 50
        texts = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # =========================
        # Step 2: 根据 ps 模拟缺失模态
        # =========================
        # ps: 缺失矩阵，形状大致是 [B, 3]
        # 3 表示三个模态：文本 language、音频 acoustic、视觉 visual
        # 例如 [1, 0, 1] 表示文本和视觉存在，音频缺失
        # generate_missing_modalities 会根据 ps 把某些模态置为缺失状态
        # 输出:
        # x_l: 缺失处理后的文本表示
        # x_a: 缺失处理后的音频表示
        # x_v: 缺失处理后的视觉表示
        x_l, x_a, x_v = self.generate_degraded_modalities(ps, texts, acoustic, visual)
        quality_l, quality_a, quality_v = self.get_modality_quality_scores(ps)
                # =========================
        # Step 3: 调整维度，适配 Conv1d
        # =========================
        # 原始形状一般是 [B, T, D]
        # Conv1d 要求输入是 [B, D, T]
        # 所以这里交换第 1 维和第 2 维
        # 这一步只是调形状，不是模型创新点
        x_l = x_l.transpose(1, 2)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
           # =========================
        # Step 4: 三个模态统一维度
        # =========================
        # 文本原始维度通常是 768
        # MOSI 音频维度是 5，视觉维度是 20
        # 不同模态维度不一样，不能直接融合
        # 所以这里用 1D 卷积把三种模态都投影到同一个维度 d_l
        # 输出形状大致是 [B, d_l, T]     
        proj_x_l = self.proj_l_1(x_l)
        proj_x_a = self.proj_a_1(x_a)
        proj_x_v = self.proj_v_1(x_v)
                # =========================
        # Step 5: 调整维度，适配 TransformerEncoder
        # =========================
        # 当前形状是 [B, d_l, T]
        # 后面的 TransformerEncoder 需要 [T, B, d_l]
        # permute(2, 0, 1) 就是把维度顺序换成:
        # 时间长度 T、batch B、隐藏维度 d_l
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        # =========================
        # Step 6: 单模态内部建模
        # =========================
        # l2l: language-to-language，文本自己内部做注意力建模
        # a2a: acoustic-to-acoustic，音频自己内部做注意力建模
        # v2v: visual-to-visual，视觉自己内部做注意力建模
        # 这里 Q/K/V 都来自同一个模态，所以是 self-attention
        # 目的：先让每个模态自己把自己的时间序列关系学好
        # interactions between ori_representations
        trans_text = self.l2l(proj_x_l,proj_x_l,proj_x_l)
        trans_text = trans_text.permute(1,2,0)
        trans_audio = self.a2a(proj_x_a,proj_x_a,proj_x_a)
        trans_audio = trans_audio.permute(1,2,0)
        trans_video = self.v2v(proj_x_v,proj_x_v,proj_x_v)
        trans_video = trans_video.permute(1,2,0)
         # =========================
        # Step 7: 池化成每个模态的整体向量
        # =========================
        # trans_text / trans_audio / trans_video 当前是序列表示
        # pool 会把一整段序列压缩成一个向量
        # last_x_l: 文本整体表示 [B, d_l]
        # last_x_a: 音频整体表示 [B, d_l]
        # last_x_v: 视觉整体表示 [B, d_l]       
        last_x_l = self.pool_text(trans_text).squeeze()# [24,30,1]
        last_x_a = self.pool_audio(trans_audio).squeeze()# [24,30,1]
        last_x_v = self.pool_video(trans_video).squeeze()# [24,30,1]
        # =========================
        # Step 8: 在当前 batch 内找相似样本做增强
        # =========================
        # get_similar_embeddings 会计算当前 batch 内样本两两之间的余弦相似度
        # 如果某些样本和当前样本相似度超过 similarity_threshold
        # 就把这些相似样本的表示取平均，作为增强信息
        # 如果找不到相似样本，使用当前 batch 内非零表示的平均值兜底
        # 这是 HME 中非常适合后续改进的位置：
        # 可以改成 top-k 检索、相似度加权、质量感知检索、记忆库检索
        # enhanced representations
        similar_texts = self.get_similar_embeddings(
            last_x_l,
            self.similarity_threshold,
            quality_scores=quality_l
        )
        similar_audios = self.get_similar_embeddings(
            last_x_a,
            self.similarity_threshold,
            quality_scores=quality_a
        )
        similar_videos = self.get_similar_embeddings(
            last_x_v,
            self.similarity_threshold,
            quality_scores=quality_v
        )
        # =========================
        # Step 9: latent 网络提炼增强表示
        # =========================
        # modality_all 把文本、音频、视觉三个整体向量堆在一起
        # 形状大致是 [B, 3, d_l]
        # latent_text/audio/video 用来进一步提炼相似样本增强信息
        # enhance_samples 是从三个模态整体表示中得到的样本级增强表示
        modality_all = torch.stack([last_x_l, last_x_a, last_x_v], dim=1)# [BSZ,DIM,DIM]
        
        eh_l = self.latent_text(similar_texts)# 
        eh_a = self.latent_audio(similar_audios)# 
        eh_v = self.latent_video(similar_videos)# 
        enhance_samples = self.latent_sample(modality_all)
                # =========================
        # Step 10: 跨模态 hyper 交互
        # =========================
        # l2a: language -> acoustic，文本信息帮助音频表示
        # l2v: language -> visual，文本信息帮助视觉表示
        # a2l: acoustic -> language，音频信息帮助文本表示
        # a2v: acoustic -> visual，音频信息帮助视觉表示
        # v2l: visual -> language，视觉信息帮助文本表示
        # v2a: visual -> acoustic，视觉信息帮助音频表示
        # 目的：让不同模态之间互相补信息，生成 hyper-modality 的基础表示
        # FUSION OF HYPERS
        hyper_l2a = self.l2a(eh_l.unsqueeze(dim=1), eh_a.unsqueeze(dim=1))
        hyper_l2v = self.l2v(eh_l.unsqueeze(dim=1), eh_v.unsqueeze(dim=1))
        hyper_a2l = self.a2l(eh_a.unsqueeze(dim=1), eh_l.unsqueeze(dim=1))
        hyper_a2v = self.a2v(eh_a.unsqueeze(dim=1), eh_v.unsqueeze(dim=1))
        hyper_v2l = self.v2l(eh_v.unsqueeze(dim=1), eh_l.unsqueeze(dim=1))
        hyper_v2a = self.v2a(eh_v.unsqueeze(dim=1), eh_a.unsqueeze(dim=1))
        # =========================
        # Step 11: 生成每个模态对应的 hyper-modality 表示
        # =========================
        # hyper_ls: 文本相关的 hyper 表示
        # hyper_as: 音频相关的 hyper 表示
        # hyper_vs: 视觉相关的 hyper 表示
        # 这些不是原始模态，而是模型根据相似样本和跨模态关系生成的增强补救表示
        # Fusions 
        hyper_ls = self.hyper_l(enhance_samples.unsqueeze(dim=1), torch.stack((hyper_l2a, hyper_l2v), dim=1))
        hyper_as = self.hyper_a(enhance_samples.unsqueeze(dim=1), torch.stack((hyper_a2l, hyper_a2v), dim=1))
        hyper_vs = self.hyper_v(enhance_samples.unsqueeze(dim=1), torch.stack((hyper_v2l, hyper_v2a), dim=1))
                # =========================
        # Step 12: VIB 过滤增强表示并估计不确定性
        # =========================
        # VIB_estimate 会输出:
        # mu: 表示分布的均值
        # std: 表示不确定性，std 越大说明越不确定
        # encoding: 经过 VIB 压缩/采样后的有效表示
        # logits: 单模态分支自己的预测结果
        # info_loss: 信息瓶颈约束损失
        # 作用：保留对情感预测有用的信息，压掉无关噪声，并告诉后面融合模块这个表示靠不靠谱
        # Use the VIB to maintain the essential information
        (mu_l_p, std_l_p), encoding_l_p, logits_l_p, info_loss_l_p = self.VIB_estimate_l_p(hyper_ls)
        (mu_a_p, std_a_p), encoding_a_p, logits_a_p, info_loss_a_p = self.VIB_estimate_a_p(hyper_as)
        (mu_v_p, std_v_p), encoding_v_p, logits_v_p, info_loss_v_p = self.VIB_estimate_v_p(hyper_vs)

        degradation_uncertainty_loss = self.compute_degradation_uncertainty_loss(
            std_l_p,
            std_a_p,
            std_v_p,
            ps
        )
                # =========================
        # Step 13: 融合三个 hyper-modality 表示
        # =========================
        # ATTN_hyper 会同时看三个模态的 encoding 和 std
        # encoding_l/a/v 是三个模态的有效增强表示
        # std_l/a/v 是三个模态的不确定性
        # 不确定性低的表示更可信，不确定性高的表示应少信
        # 输出 hyper_all: 总的 hyper 增强表示
        # Fuse the hyper-modalities
        hyper_all = self.ATTN_hyper(encoding_l_p, encoding_a_p, encoding_v_p, std_l_p, std_a_p, std_v_p)
                # =========================
        # Step 14: 把 hyper 增强信息融合回原始模态
        # =========================
        # refine_ls: 原始文本表示 + 文本 hyper 表示 + 文本不确定性 得到的修正文本表示
        # refine_as: 原始音频表示 + 音频 hyper 表示 + 音频不确定性 得到的修正音频表示
        # refine_vs: 原始视觉表示 + 视觉 hyper 表示 + 视觉不确定性 得到的修正视觉表示
        # LB / UB 是不确定性上下界
        # 这是后续做“质量感知可靠融合”的重要改进位置
        # Fuse the hyper-modalities into ori_representations
        refine_ls = self.ATTN_T(last_x_l, encoding_l_p, std_l_p, self.LB, self.UB)
        refine_as = self.ATTN_A(last_x_a, encoding_a_p, std_a_p, self.LB, self.UB)
        refine_vs = self.ATTN_V(last_x_v, encoding_v_p, std_v_p, self.LB, self.UB)
        # =========================
        # Step 15: 最终融合并预测情感分数
        # =========================
        # all_embed 包含 4 路信息:
        # 1. 修正后的文本 refine_ls
        # 2. 修正后的音频 refine_as
        # 3. 修正后的视觉 refine_vs
        # 4. 总 hyper 表示 hyper_all
        # transformer_encoder_all 让这 4 路信息再次交互
        # cat 后得到 [B, 4*d_l]
        # fusion_all 是最终 MLP 预测头，输出情感分数 output: [B, 1]
        all_embed = torch.stack((refine_ls, refine_as, refine_vs, hyper_all), dim=0)
        hidden = self.transformer_encoder_all(all_embed)
        hidden = torch.cat((hidden[0], hidden[1], hidden[2], hidden[3]), dim=1)
        output = self.fusion_all(hidden)
                # =========================
        # Step 16: 整理输出
        # =========================
        # logits_l_p/a_p/v_p: 三个单模态辅助预测，用于辅助 loss
        # predictions: 最终多模态情感预测，用于主任务 loss 和测试指标
        # info_losses: VIB 信息瓶颈损失，会在 HME_main.py 中加入总 loss
        info_losses = (info_loss_l_p + info_loss_a_p + info_loss_v_p) / 3.0
        res = {
                "logits_l_p": logits_l_p,
                "logits_a_p": logits_a_p,
                "logits_v_p": logits_v_p,
                "predictions": output,
                "info_losses": info_losses,
                "degradation_uncertainty_loss": degradation_uncertainty_loss,
        }

        return res
       