# 03 HME-TopK Results

这里只记录 HME-TopK。

| run_id | dataset | missing_rate | top_k | temperature | seed | epochs | train_loss | valid_loss | mae | corr | acc7 | acc5 | acc2_non_zero | f1_non_zero | acc2 | f1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sanity_mosi_topk_mr02_k3_t01_s5576_e1 | MOSI | 0.2 | 3 | 0.1 | 5576 | 1 | 2.3709 | 2.4685 | 1.2490 | 0.6121 | 0.1764 | 0.1764 | 0.7607 | 0.7602 | 0.7434 | 0.7420 | sanity check only |
| pilot_mosi_topk_mr02_k3_t01_s5576_e20 | MOSI | 0.2 | 3 | 0.1 | 5576 | 20 | 1.3887 | 1.8807 | 0.8630 | 0.7094 | 0.4067 | 0.4519 | 0.7896 | 0.7902 | 0.7755 | 0.7754 | pilot comparison on MOSI mr=0.2 seed=5576 |
| pilot_mosi_topk_mr00_k3_t01_s5576_e20 | MOSI | 0.0 | 3 | 0.1 | 5576 | 20 | 0.5768 | 1.4057 | 0.7192 | 0.7960 | 0.4636 | 0.5262 | 0.8506 | 0.8483 | 0.8251 | 0.8215 | pilot comparison on MOSI multi missing rates seed=5576 |
| pilot_mosi_topk_mr01_k3_t01_s5576_e20 | MOSI | 0.1 | 3 | 0.1 | 5576 | 20 | 0.9421 | 1.6049 | 0.8320 | 0.7162 | 0.4082 | 0.4519 | 0.8003 | 0.8012 | 0.7813 | 0.7815 | pilot comparison on MOSI multi missing rates seed=5576 |
| pilot_mosi_topk_mr03_k3_t01_s5576_e20 | MOSI | 0.3 | 3 | 0.1 | 5576 | 20 | 1.6626 | 2.0505 | 0.9075 | 0.6828 | 0.3790 | 0.4300 | 0.7729 | 0.7710 | 0.7595 | 0.7569 | pilot comparison on MOSI multi missing rates seed=5576 |
| pilot_mosi_topk_mr05_k3_t01_s5576_e20 | MOSI | 0.5 | 3 | 0.1 | 5576 | 20 | 1.9192 | 2.2512 | 1.0660 | 0.5709 | 0.3076 | 0.3469 | 0.7149 | 0.7114 | 0.6997 | 0.6950 | pilot comparison on MOSI multi missing rates seed=5576 |
| pilot_mosi_topk_mr02_k3_t01_s1111_e20 | MOSI | 0.2 | 3 | 0.1 | 1111 | 20 | 0.9858 | 1.6619 | 0.8428 | 0.7125 | 0.3892 | 0.4461 | 0.8079 | 0.8071 | 0.7886 | 0.7869 | pilot multi-seed check on MOSI seed=1111 |
| pilot_mosi_topk_mr03_k3_t01_s1111_e20 | MOSI | 0.3 | 3 | 0.1 | 1111 | 20 | 1.2532 | 1.8351 | 0.9540 | 0.6336 | 0.3776 | 0.4286 | 0.7530 | 0.7503 | 0.7347 | 0.7307 | pilot multi-seed check on MOSI seed=1111 |
| pilot_mosi_topk_mr05_k3_t01_s1111_e20 | MOSI | 0.5 | 3 | 0.1 | 1111 | 20 | 1.8610 | 2.1774 | 1.1666 | 0.5059 | 0.2741 | 0.3061 | 0.6357 | 0.6256 | 0.6385 | 0.6271 | pilot multi-seed check on MOSI seed=1111 |
| pilot_mosi_topk_mr02_k3_t01_s2222_e20 | MOSI | 0.2 | 3 | 0.1 | 2222 | 20 | 0.9476 | 1.5838 | 0.8850 | 0.6939 | 0.3994 | 0.4329 | 0.7851 | 0.7860 | 0.7711 | 0.7714 | pilot multi-seed check on MOSI seed=2222 |
| pilot_mosi_topk_mr03_k3_t01_s2222_e20 | MOSI | 0.3 | 3 | 0.1 | 2222 | 20 | 1.4196 | 1.9182 | 0.8922 | 0.6826 | 0.3921 | 0.4461 | 0.7729 | 0.7738 | 0.7609 | 0.7612 | pilot multi-seed check on MOSI seed=2222 |
| pilot_mosi_topk_mr05_k3_t01_s2222_e20 | MOSI | 0.5 | 3 | 0.1 | 2222 | 20 | 1.8770 | 2.1657 | 1.1159 | 0.5272 | 0.2901 | 0.3134 | 0.7149 | 0.7162 | 0.7041 | 0.7044 | pilot multi-seed check on MOSI seed=2222 |
| official_mosi_topk_mr05_k3_t01_s5576_e100 | MOSI | 0.5 | 3 | 0.1 | 5576 | 100 | 1.2917 | 1.9127 | 1.0938 | 0.5083 | 0.3105 | 0.3382 | 0.7088 | 0.7095 | 0.6953 | 0.6950 | official check on MOSI mr=0.5 seed=5576 epochs=100 |
| official_mosi_topk_mr05_k3_t01_s1111_e100 | MOSI | 0.5 | 3 | 0.1 | 1111 | 100 | 1.2093 | 1.7607 | 1.0814 | 0.5431 | 0.3120 | 0.3426 | 0.6845 | 0.6863 | 0.6778 | 0.6787 | official MOSI mr=0.5 multi-seed check |
| official_mosi_topk_mr05_k3_t01_s2222_e100 | MOSI | 0.5 | 3 | 0.1 | 2222 | 100 | 1.3906 | 1.9977 | 1.0743 | 0.5354 | 0.3134 | 0.3528 | 0.6997 | 0.6988 | 0.6837 | 0.6815 | official MOSI mr=0.5 multi-seed check |
