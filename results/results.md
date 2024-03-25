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



# beta1&2筛选

![RAF-DB_CLSFERBaseline_SearchBETA1](C:\Users\wangj\Downloads\RAF-DB_CLSFERBaseline_SearchBETA1.png)

（0.9, 0.999)就行

# 长epoch测试（50~60）

大LR导致LOSS变NaN情况明显，需要进一步降低LR/考虑正则方法。

在长epoch、较低LR条件下，重新验证depth影响，更深的模型表现出更好的效果。

![CLSFERBaseline_SearchDepth&LR](D:\College\projects\Test Models\results\CLSFERBaseline_SearchDepth&LR.png)

> 指标通常对LR高度敏感，因此大部分调参情况下，都需要将LR作为冗余参数共同调参。
>
> depth的性能需要较长的epoch、合适的LR才能够充分体现。目前来看需要做grid search

同时在depth8的条件下，验证eps的效果；实验结果无法说明eps和指标有高相关性，考虑将其作为固定参数，暂时设置为1e-8

![CLSFERBaseline_eps](D:\College\projects\Test Models\results\CLSFERBaseline_eps.png)

> 整理调参工作流：
>
> 1. 参照以往论文和调参经验，设置合适的初始参数。
> 2. 适当验证optimizer参数，影响不明显
> 3. 较长epoch（50~60）验证模型架构，对不同lr进行grid search
> 4. 验证weight decay



# weight decay

试算了两种decay，效果不是很理想，考虑到目前要调参的项还很多，暂时还是先不用weight decay吧。

![CLSFERBaseline_scheduler](D:\College\projects\Test Models\results\CLSFERBaseline_scheduler.png)



# CLS ViT 验证

