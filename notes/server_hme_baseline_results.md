<!--
 * @Author: Leo
 * @Date: 2026-05-25 16:56:47
 * @LastEditTime: 2026-05-25 16:57:02
 * @LastEditors: Leo
 * @Description: 
-->
# HME 服务器原始 baseline 实验记录

## 1. 实验目的

在服务器上复现原始 HME 代码，得到原始 baseline 结果，作为后续改进方法的对比基准。

## 2. 代码版本

- 方法：原始 HME
- 数据集：CMU-MOSI
- 模态：文本 / 音频 / 视觉
- 任务：缺失模态多模态情感分析
- 是否改代码：否，只加了注释，不改变模型逻辑

## 3. 服务器环境

- GPU：
- 显存：
- CUDA：
- Python：
- PyTorch：
- transformers：
- sentence-transformers：

## 4. 数据设置

- 数据集：MOSI
- 数据文件：datasets/aligned_mosi.pkl
- 缺失率 missing_rate：
- seed：

## 5. 运行命令

```bash
python ./HME_MSA/HME_main.py \
  --dataset='mosi' \
  --learning_rate=2e-5 \
  --d_l=192 \
  --missing_rate=0.2 \
  --layers=4 \
  --hyper_depth=3 \
  --latent_layers=4 \
  --latent_dim=192 \
  --num_latents=5 \
  --train_batch_size=256 \
  --dev_batch_size=128 \
  --test_batch_size=128 \
  --n_epochs=100