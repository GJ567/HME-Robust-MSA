import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from numpy.random import randint


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
 
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# 根据 ps 对三个模态进行缺失处理
# =========================
# 作用：
#   根据 generate_missing_matrix 生成的 ps，
#   把文本、音频、视觉中被标记为缺失的模态处理成缺失状态。
#
# 输入：
#   ps: 缺失矩阵，形状大致是 [B, 3]
#   texts: BERT 编码后的文本表示
#   acoustic: 音频特征
#   visual: 视觉特征
#
# 输出：
#   x_l: 缺失处理后的文本表示
#   x_a: 缺失处理后的音频表示
#   x_v: 缺失处理后的视觉表示
#
# 大白话：
#   ps 告诉模型“这一条样本哪些模态没了”，
#   这个函数真正执行“把对应模态变成缺失”的操作。
#
# 后续可改方向：
#   1. 缺失模态直接置零
#   2. 低质量模态加噪声
#   3. 文本随机 mask token
#   4. 音频加入高斯噪声
#   5. 视觉特征随机遮挡或扰动
#   6. 根据质量分数决定是置零、降权还是加噪

def generate_missing_modalities(mask, text_features, audio_features, video_features):
    # "Mask should have shape [batchsize, 3]"
    assert mask.shape[1] == 3 
    assert text_features.shape[0] == mask.shape[0]
    assert audio_features.shape[0] == mask.shape[0]
    assert video_features.shape[0] == mask.shape[0]
    zero_vector_text = torch.zeros_like(text_features)
    zero_vector_audio = torch.zeros_like(audio_features)
    zero_vector_video = torch.zeros_like(video_features)
    mask = mask.bool()
    
    masked_text_features = torch.where(mask[:, 0].unsqueeze(1).unsqueeze(2), text_features, zero_vector_text)
    masked_audio_features = torch.where(mask[:, 1].unsqueeze(1).unsqueeze(2), audio_features, zero_vector_audio)
    masked_video_features = torch.where(mask[:, 2].unsqueeze(1).unsqueeze(2), video_features, zero_vector_video)
    return masked_text_features, masked_audio_features, masked_video_features
# =========================
# 生成缺失模态矩阵 ps
# =========================
# 作用：
#   根据 missing_rate 随机生成每个样本的模态缺失情况。
#
# 输入：
#   nums: 当前数据集样本数，比如 MOSI train 是 1284
#   modality_num: 模态数量，这里通常是 3，表示文本/音频/视觉
#   missing_rate: 缺失率，比如 0.2 表示约 20% 的模态被设为缺失
#
# 输出：
#   ps: 缺失矩阵，形状大致是 [nums, 3]
#   每一行对应一个样本，每一列对应一个模态
#   1 表示该模态存在，0 表示该模态缺失
#
# 例子：
#   [1, 1, 1] 表示文本、音频、视觉都存在
#   [1, 0, 1] 表示音频缺失
#   [0, 1, 1] 表示文本缺失
#
# 后续可改方向：
#   1. 随机缺失 -> 质量驱动缺失
#   2. 只模拟缺失 -> 同时模拟缺失、噪声、低质量
#   3. 固定 missing_rate -> 根据模态质量动态生成缺失概率
def generate_missing_matrix(M, view_num=3, missing_rate=0.5):
    """
    generate a mask matrix whith shape [M, view_num], and keep that at least one modality exist in each sample
    missing rate is the missing modalities in all modalities
    """
    one_rate = 1 - missing_rate
    
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(M, 1))).toarray()
        return view_preserve
    
    if one_rate == 1: 
        matrix = randint(1, 2, size=(M, view_num))
        return matrix
    
    alldata_len = max(M, 32)
    error = 1
    while error >= 0.005:
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int32)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int32))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int32)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int32)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    
    matrix = matrix[:M, :]
    return matrix


def modality_drop(modal1, modal2, modal3, p, args):
    # The code is adapted from MMANet: https://github.com/shicaiwei123/MMANet-CVPR2023/blob/main/classification/lib/model_arch.py
    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    index_list = [x for x in range(7)]
    # for training with uncertain conditions
    # save ps for uncertain modalities reweighting
    if p == [0, 0, 0]:
        p = []
        prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
        for i in range(modal1.shape[0]):# bsz
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p.append(modality_combination[index])# store each missing conditions
            
        p = np.array(p)
        p = torch.from_numpy(p)# adds the missing conditions into the list
        p1 = torch.unsqueeze(p, 2)
        p1 = torch.unsqueeze(p1, 3)
        # p1 = torch.unsqueeze(p1, 4)
    # for testing with fixed conditions
    else:
        p = p
        p = [p * modal1.shape[0]]
        p = np.array(p).reshape(modal1.shape[0], 3)# for 3 modalities
        p = torch.from_numpy(p)
        p1 = torch.unsqueeze(p, 2)
        p1 = torch.unsqueeze(p1, 3)
 
    p1 = p1.float().to(modal1.device)#.to(DEVICE)
    p = p.float().to(modal1.device)#.to(DEVICE)

    modal1 = modal1 * p1[:, 0].to(modal1.device)
    modal2 = modal2 * p1[:, 1].to(modal2.device)
    modal3 = modal3 * p1[:, 2].to(modal3.device)

    return modal1, modal2, modal3, p


