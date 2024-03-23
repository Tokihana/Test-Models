# Baseline(IR50 + Vanilla ViT Block)

batch size选择

| batch size | throughput | time per step |
| ---------- | ---------- | ------------- |
| 1          | 125        | 0.008         |
| 2          | 248        | 0.008         |
| 4          | 509        | 0.008         |
| 8          | 959        | 0.008         |
| 16         | 1279       | 0.012         |
| 32         | 1359       | 0.024         |
| 64         | 1449       | 0.044         |
| 128        | 1543       | 0.083         |
| 256        | 1591       | 0.161         |
| 512        | 1602       | 0.319         |

用256应该更好一些（depth6和depth8也可行）。

对depth为4的baseline，LR的调参结果如下

![baseline_depth4_SearchLR](D:\College\projects\Test Models\results\baseline_depth4_SearchLR.png)

基本上2e-4左右就好。

depth6，区间也在[1e-5, 1e-3]

![CLSFERBaseline_depth8_SearchLR](D:\College\projects\Test Models\results\CLSFERBaseline_depth8_SearchLR.png)

depth8，区间基本一致

![RAF-DB_CLSFERBaseline_depth8_SearchLR](D:\College\projects\Test Models\results\RAF-DB_CLSFERBaseline_depth8_SearchLR.png)

不同depth（2, 4, 6, 8）的acc对比（确定模型层数）

配置好实验名称，方便筛选

```
CLSFERBaseline_depth2_SearchLR
CLSFERBaseline_depth4_SearchLR
CLSFERBaseline_depth6_SearchLR
CLSFERBaseline_depth8_SearchLR
```

![CLSFERBaseline_depth_test](D:\College\projects\Test Models\results\CLSFERBaseline_depth_test.png)

这样来看，2depth可能足够了，LR取个4e-4

# eps筛选

![RAF-DB_CLSFERBaseline_SearchEPS](D:\College\projects\Test Models\results\RAF-DB_CLSFERBaseline_SearchEPS.png)

对depth2的模型来说，1e-7更合适些。



# beta1筛选

![RAF-DB_CLSFERBaseline_SearchBETA1](C:\Users\wangj\Downloads\RAF-DB_CLSFERBaseline_SearchBETA1.png)

0.9就行

