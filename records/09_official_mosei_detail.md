# 09 Official MOSEI Detail

MOSEI official 泛化实验明细。当前只包含 seed=5576，mr=0.2 / 0.5，HME 和 DARE-HME。

| run_id | dataset | missing_rate | model_version | seed | epochs | mae | corr | acc2 | f1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_mosei_hme_mr02_s5576_e100 | MOSEI | 0.2 | HME | 5576 | 100 | 0.6011 | 0.6755 | 0.7839 | 0.7859 | official MOSEI mr=0.2 seed=5576 generalization check |
| official_mosei_dare_mr02_s5576_e100 | MOSEI | 0.2 | DARE-HME | 5576 | 100 | 0.545 | 0.7538 | 0.8036 | 0.8088 | official MOSEI mr=0.2 seed=5576 generalization check |
| official_mosei_hme_mr05_s5576_e100 | MOSEI | 0.5 | HME | 5576 | 100 | 0.6833 | 0.5402 | 0.7334 | 0.7312 | official MOSEI mr=0.5 seed=5576 generalization check |
| official_mosei_dare_mr05_s5576_e100 | MOSEI | 0.5 | DARE-HME | 5576 | 100 | 0.5661 | 0.7308 | 0.8049 | 0.8081 | official MOSEI mr=0.5 seed=5576 generalization check |
