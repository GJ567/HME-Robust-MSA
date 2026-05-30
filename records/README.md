# 实验记录说明

## 文件说明

- 00_environment.md  
  记录服务器环境、Python/PyTorch/transformers 版本、数据集信息。

- 01_run_index.csv  
  所有实验运行的总目录。每跑一次实验都要加一行。

- 02_hme_baseline_results.csv  
  只记录原始 HME baseline 的结果。

- 03_hme_topk_results.csv  
  只记录 HME-TopK 的结果。

- 04_dare_hme_results.csv  
  只记录 DARE-HME 完整模型的结果。

- 05_ablation_results.csv  
  记录消融实验结果。

- 06_final_summary.csv  
  最终论文汇总表，等多个 seed 跑完后再填写 mean ± std。

## 记录原则

1. sanity check 只证明代码能跑，不作为论文正式结果。
2. 正式结果至少跑多个 missing_rate。
3. 关键实验至少跑 3 个 seed。
4. 不同模型版本不要混在同一张表里。
5. 每次运行必须有唯一 run_id。
