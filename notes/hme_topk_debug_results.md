<!--
 * @Author: Leo
 * @Date: 2026-05-26 15:18:18
 * @LastEditTime: 2026-05-26 15:18:30
 * @LastEditors: Leo
 * @Description: 
-->
# HME-TopK Debug Result

## 时间
2026-05-26

## 环境
conda env: hme38

## 命令
```bash
python HME_MSA/HME_main_topk.py --dataset mosi --n_epochs 1 --train_batch_size 16 --dev_batch_size 16 --test_batch_size 16 --top_k 3 --top_temperature 0.1


结果

代码成功跑通 1 个 epoch。

train_loss: 4.2223
valid_loss: 2.6139
test_acc2: 0.4534
mae: 1.4652
corr: 0.0581
acc7: 0.1531
acc5: 0.1531
f_score: 0.3014



---

## 然后再跑正式一点的版本

下一步可以把 batch 改回正常一点，先跑 MOSI：

```bash
python HME_MSA/HME_main_topk.py --dataset mosi --n_epochs 100 --train_batch_size 256 --dev_batch_size 128 --test_batch_size 128 --top_k 3 --top_temperature 0.1