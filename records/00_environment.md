# HME / HME-TopK / DARE-HME 实验环境记录

## Server
- Platform: SCNet
- GPU: NVIDIA GeForce RTX 4090
- Project path: /root/private_data/HME/HME

## Conda Environment
- Env name: hme38
- Python: 3.8
- torch: 1.12.1+cu113
- CUDA: 11.3
- transformers: 4.30.2
- numpy: 1.24.4
- sklearn: 1.3.2
- scipy: 1.10.1
- einops: 0.8.1

## Important Note
旧服务器环境 torch 2.4.1 + transformers 4.46.3 会导致 HME 中 LayerNorm 出现 NaN。
正式实验统一使用当前 hme38 环境。

## Dataset
- MOSI file: datasets/aligned_mosi.pkl
- Current format: HME list format
- Required keys: train / dev / test
