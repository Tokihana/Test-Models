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

## eps筛选

![RAF-DB_CLSFERBaseline_SearchEPS](D:\College\projects\Test Models\results\RAF-DB_CLSFERBaseline_SearchEPS.png)

对depth2的模型来说，1e-7更合适些。



## beta1&2筛选

![RAF-DB_CLSFERBaseline_SearchBETA1](C:\Users\wangj\Downloads\RAF-DB_CLSFERBaseline_SearchBETA1.png)

（0.9, 0.999)就行

## 长epoch测试（50~60）

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



## weight decay

试算了两种decay，效果不是很理想，考虑到目前要调参的项还很多，暂时还是先不用weight decay吧。

![CLSFERBaseline_scheduler](D:\College\projects\Test Models\results\CLSFERBaseline_scheduler.png)



# CLS ViT 验证

![CLSFER_Compare](D:\College\projects\Test Models\results\CLSFER_Compare.png)

| LR       | MODEL    | ACC        |
| -------- | -------- | ---------- |
| 3.20E-05 | Baseline | 82.986     |
|          | CLSFER   | **83.638** |
| 1.80E-05 | Baseline | 84.452     |
|          | CLSFER   | **84.778** |
| 1.00E-05 | Baseline | 83.67      |
|          | CLSFER   | **84.387** |
| 5.60E-06 | Baseline | 84.713     |
|          | CLSFER   | **85.039** |
| 3.20E-06 | Baseline | 83.605     |
|          | CLSFER   | **85.398** |

为什么用CLSBlock效果更好一些？



## 运行效率验证

| Model                      | Batch Size | throughput | FLOPs(G)  | Params.(M) | Memory Usage(M) | Memory Usage(G) |
| -------------------------- | ---------- | ---------- | --------- | ---------- | --------------- | --------------- |
| Baseline(IR50 + ViT Block) | 256        | 1381       | 7.594     | **55.954** | 21580           | 21.07           |
| IR50 + CLS  Block          | 256        | **1790**   | **6.385** | 55.955     | **18479**       | 18.05           |



## Baseline取点重复实验

| LR       | Model                      | Acc   |
| -------- | -------------------------- | ----- |
| 5.60E-06 | Baseline(IR50 + ViT Block) | 84.55 |



## 测试下在CPU模式下的通过速度

本机的CPU为13th Gen Intel(R) i9-13900HX

| throughput | time per step |
| ---------- | ------------- |
| 1          | 2.88          |



## Mixup alpah 验证

在[0, 1]范围内，趋近1效果好像更好一些，调个2测试一下，如果效果变差的话，说明维持在均匀一些的分布（alpha = 1）可能更好。目前来看，模型对LR的敏感度更高，可以的话还是加一些正则化的方法来降低敏感度吧。

这个测试结果和MixAug的结论有很大差别，MixAug的测试结果是0.1更好；当然，MixAug测试的是ResNet50，模型结构变了结果发生变化也正常。

还有一种可能性是，我只用的CrossEntroy，是不是需要用SoftTarget或者LabelSmoothing？

![CLSFER_NonMulti_Mixup](D:\College\projects\Test Models\results\CLSFER_NonMulti_Mixup.png)



## depth 验证

CLS FER这边没做过depth + LR的验证，测试一下吧。

![CLSFER_NonMulti_SearchDepth](D:\College\projects\Test Models\results\CLSFER_NonMulti_SearchDepth.png)

整体上来看，还是深一点表现更好。那就还是用depth 8



## stage验证

尝试用一下stage3的特征，与stage4的特征进行对比。





## Weight decay验证

上次weight decay的效果并不是很理想，我认为可能是weight decay和LR关系比较大的原因，所以这次将decay和LR一起调参？



## drop attn验证









# 下一步工作

目前的实验结果，能够支撑CLS Block的可行性，我们可以适当画个图来说明这个结构。但为啥效果反而高了一些？

下一步肯定是要向SOTA靠拢的，首先要再次确定下SOTA的范围，跑一部分有源码的SOTA，比较一下效果？

但怎么在CLS Block的基础上进一步改进有点没头绪，目前已经知道的可能包括：Multi Scale，加landmark，进一步压缩模型。

我觉得首先还是得先往SOTA的表现靠拢，然后再考虑下一步压缩？

嗯，那就按下面这个顺序来处理吧：

1. 列SOTA表，找一下POSTER之后，23年有没有进一步提升表现的论文（有源码优先）
2. 梳理这些SOTA的结构，考虑我们架构如果向这些SOTA靠拢的话，能不能区分出模型差异。
3. 寻找近两年的新hybrid ViT架构，尝试将CLS Token融合进去。

从另一方面来说，当前面临的问题还是属于解释角度的问题，需要找到一个不错的解释角度来向SOTA架构靠拢。

我需要一轮新的知识突破才能打破目前的困境，当前的困境很大程度上还是来自于对SOTA的发展路径、性能提升点不够熟悉，以及对各种trick的了解不够充分，得进一步加深对FER领域和模型架构的理解才行。



## SOTA 表





