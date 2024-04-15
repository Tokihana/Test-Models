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

参照POSTER的实现，从[14, 14, 256]的特征图直接view到[49, 1024]的特征图

![CLSFER_NonMulti_Stage3](D:\College\projects\Test Models\results\CLSFER_NonMulti_Stage3.png)

从结果上来看，使用stage3的效果会更好一些。



## Weight decay验证

上次weight decay的效果并不是很理想，我认为可能是weight decay和LR关系比较大的原因，所以这次将decay和LR一起调参？

目前改用了reduce on plateau，从观察来看，decay曲线确实比较接近exponential。

调整初始LR 对模型效果确实会存在影响，也许需要跑一个实验来找到合适的组合。



## drop attn验证

drop attn的效果不是很稳定的样子，后面需要提精度的时候可以尝试用一用

![CLSFER_NonMulti_Stage3_attndrop](D:\College\projects\Test Models\results\CLSFER_NonMulti_Stage3_attndrop.png)







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



## 

# 组会讨论

ViT的论文有没有说明CLS Token和feature token具体包含什么信息？有没有说过。

FER领域，表情相关和表情无关信息最早是谁提的。



1. 其他数据集上也出现这种情况，证明结果有效性。
2. 传统ViT会分CLS 和feature，分别代表什么信息。-> ViT最开始的论文，立论基础
3. CLS token反映了哪些信息，feature token反映了哪些信息。-> feature token**对表情识别没有用**，**该怎么证明**。
4. 精度下一步可以做着用。



做哪些实验：

1. baseline&CLS在其他数据集->证明现象是普遍的。
2. 如果证明是普遍现象，想办法证明feature token没有用。**修正：不需要证明没有用**，因为现有实验证据无法说明没有用（Baseline还是比单用CLS token高一些的），所以只需要论证CLS token已经足够使用即可。



# FERPlus测试

首先修正了FERPlus的转存方法，之前存的是png，位深32，FERPlus本来就是灰度图，所以转存成jpg了。

对应的文件为FERplus_split.py

由于Baseline_Stage3跑不了256的batch size，调整为224，**后面发现是embed_dim的问题，设置1024有些太大了，POSTER中是做了一次linear映射，我们在这里直接保留了resnet layer4的第一个降维block**

顺带调整了输入的img size为112，SOTA也是直接interploate到112的。主要是FERPlus数据本来就不大（48x48），先升维再降维不合理。这样的话，ir50中的conv1就不用设stride2了，改回stride1。

FERPlus的测试时间确实比较长，单个epoch要1min40s。连带上LR测试的话，估计要跑8h左右。

![FERPlus_SearchLR&Arch](D:\College\projects\Test Models\results\FERPlus_SearchLR&Arch.png)

确认了一下测试时间太长的原因：stage3不应使用view，会占用大量开销；

已经调整了stage3的实现，修正为使用ir50自己的block，只不过把layer4的深度调整为1，只留一层降维。



# 增加Reduce LR，跑个200epoch

对比4种架构：NonMulti, Baseline, NonMulti_stage3, Baseline_stage3

![RAF-DB_200epoch_reduce](D:\College\projects\Test Models\results\RAF-DB_200epoch_reduce.png)

可以看出来，使用CLS的话，精度还是会差一些的，另外stage3确实效果更优；这里没用mixup，用了的话效果可能会更好一些。

| arch            | acc    |
| --------------- | ------ |
| Baseline_stage3 | 87.321 |
| NonMulti_stage3 | 87.256 |
| Baseline_stage4 | 85.724 |
| NonMulti_stage4 | 85.494 |



# FERPlus跑200epoch

有点困难，Baseline的loss经常会变NaN，目前已经将初始学习率折半两次了。但CLS block的实现没这个现象。这是不是也是可以寻找解释的？

> 从目前的经验来看，一种可行的解释方向是，排除了feature token的干扰后，虽然会有一定程度的信息损失，但模型的收敛性得到了提升。
>
> 有没有办法令鱼与熊掌兼得？即前半部分保持用feature token，后半部分使用CLS token？
>
> 下一步如果做AffectNet的运行测试，估计还是得继续降LR

另一方面，从目前保有的实验结果来看，如果loss没有NaN的话，那么Baseline的结果是会好一些的。

学习率折半两次还是会变NaN，但每次折半变NaN的时间都会延后一些。从目前实验结果来看，Baseline会好一些。

![FERPlus_200epoch](D:\College\projects\Test Models\results\FERPlus_200epoch.png)

对于变NaN这个问题，目前已知的信息：

- 不是初始epoch就变，而是在训练several epoch后，突然跳变为NaN
- 调小学习率能够延缓变NaN的epoch数

我怀疑是timm的LayerNorm实现不太行，所以改了nn.LayerNorm，到时再测测吧



# 把baseline的NaN debug一下

1. 改LayerNorm为nn.LayerNorm，还是会NaN

   但是通过两次的对比，观察在相同LR下，NaN会在固定epoch的特定log位置出现。高度怀疑是发生在同一位置

2. 启用baseline的qk_norm

   这个方法似乎是有效的，等跑完200epoch看看结果吧。预计还需要再跑个2.5e-5学习率的结果。

   从qk_norm原论文来看，在qk上加norm能够降低softmax饱和的倾向，减少梯度问题。

3. 自己实现一遍block？

如果qk_norm后面没问题的话，在2.5e-5下再跑一次两个stage3的对照试验。跑完就开始AffectNet的部分。

结果：

| lr     | Baseline | CLS    |
| ------ | -------- | ------ |
| 1e-4   | 82.212   | 81.368 |
| 2.5e-5 | 83.76    | 82.944 |



# 跑个AffectNetLR

因为跑的很慢，所以首先10epoch确定了一个小的LR范围，目前来看1e-5和1e-6都还不错，后续可能还是会用1e-6，因为FERPlus在2.5e-5效果更好一些，AffectNet肯定是要再小一点的。

已经跑上了，估计得跑一整天。

过拟合很明显，停了，加个attn_drop。

0.4的attn_drop稍微缓解了一点过拟合的情况，另外，好像很接近SOTA的baseline了？

下一步要做的实验：

- 区分一下training loss和val loss，方便判断拟合情况
- 继续添加正则化方法，降低过拟合影响。
  - 先加个0.2的mixup
  - 加0.2的proj_drop
  - 改用exponential
- 经过上述正则化后，确实一定程度上抑制了过拟合，但过拟合现象还是很严重。
- 去掉attn_drop和proj_drop，改用0.5的drop_path，效果会更稳定一些

目前来看，过拟合还是很严重，后面再想想办法吧。



# RepeatCLS

现在有这样一个想法，设原始的attn计算为
$$
\left[\begin{array}{}cls\\patch\end{array}\right] @
\left[\begin{array}{}cls & patch\end{array}\right] \to \left[\begin{array}{}cls\cdot cls& cls \cdot patch\\ patch\cdot cls & patch \cdot patch \end{array}\right]\\
\left[\begin{array}{}cls\cdot cls& cls \cdot patch\\ patch\cdot cls & patch \cdot patch \end{array}\right] @ \left[\begin{array}{}cls\\patch\end{array}\right] \to \left[\begin{array}{}cls^3 + cls\cdot patch \cdot cls\\
patch\cdot cls \cdot cls + patch^3\end{array}\right]\\
\left[\begin{array}{}cls^3 + cls\cdot patch \cdot cls\\
patch\cdot cls \cdot cls + patch^3\end{array}\right] + \left[\begin{array}{}cls\\patch\end{array}\right]
$$
 而对于Cross，其计算为
$$
\left[\begin{array}{}cls\end{array}\right] @
\left[\begin{array}{}cls & patch\end{array}\right] \to \left[\begin{array}{}cls\cdot cls& cls \cdot patch\end{array}\right]\\
\left[\begin{array}{}cls\cdot cls& cls \cdot patch \end{array}\right] @ \left[\begin{array}{}cls\\patch\end{array}\right] \to \left[\begin{array}{}cls^3 + cls\cdot patch \cdot cls\end{array}\right]\\
\left[\begin{array}{}cls^3 + cls\cdot patch \cdot cls\end{array}\right] + \left[\begin{array}{}cls\end{array}\right]
$$
在我们的Cross实现中，始终没有更新patch，是否可以通过将cls repeat，加到patch上，对其进行更新？

- 在attn之后就repeat，参与v的计算（NN开销仍然存在，丧失Cross的速度优势）
- 在CLS计算v之后repeat，将更新后的cls加到patch上（运算开销增大，运行效率优势丧失$\frac 2 3$，但确实有补偿效果）

$$
\left[\begin{array}{}cls^3 + cls\cdot patch \cdot cls\\ N \times(cls^3 + cls\cdot patch \cdot cls)\end{array}\right] + \left[\begin{array}{}cls\\patch\end{array}\right]
$$

后一个思路目前用的repeat，可能是导致开销变大的原因，因为数据不需要复制一份，改用expand试一下吧（会高一点，但高的不多）。

| arch     | throughput | flops(G)  | params.(M) |
| -------- | ---------- | --------- | ---------- |
| Baseline | 1410       | 7.133     | 46.512     |
| CLSFER   | **1882**   | **5.922** | **46.510** |
| Repeat   | 1496       | 6.925     | **46.510** |
| Expand   | 1499       | 6.925     | **46.510** |

还是再找找有没有更好的思路吧。



# 好像出现了bug

今天实现MultiScale的时候，发现NonMulti的实现中，更新cls token的位置，似乎存在代码错误

错误代码

```
x = x[:, 0:1, ...] + self.attn_drop(self.attn(self.norm1(x)))
```

理论上正确的代码

```
x[:, 0:1, ...] = x[:, 0:1, ...] + self.attn_drop(self.attn(self.norm1(x)))
```

错误代码会使得整个x的shape变为(B, 1, C)，而正确的形式应该为(B, N, C)，由于错误代码会直接丢掉patch部分，且在运行中没有保存，因此需要重新跑一轮实验，确定错误代码的影响。

由于直接赋值给`x[:, 0:1, ...]`，会导致`.backward()`报错`inplace error`，因此需要赋值给一个新的变量。

```
x_cls = x[:, 0:1, ...] + self.attn_drop(self.attn(self.norm1(x)))
new_x = torch.cat((x_cls, x[:, 1:, ...].clone()), dim=1)
```

cat操作可能会导致运行效率下降，我需要测一测。

先加个正则化，loss抖得太厉害了。Mixup0.2, stochastic drop path 0.5

试着把cat放在mlp后面了，性能改善了很多。

| arch            | throughput | FLOPs(G) | Params.(M) |
| --------------- | ---------- | -------- | ---------- |
| NonMultiCLS_cat | 1487       | 6.925    | 46.510     |
| CLS_catAfterMlp | 1782       | 6.102    | 46.510     |
| CLS_only        | 1886       | 5.922    | 46.510     |
| Baseline        | 1400       | 7.133    | 46.512     |

抽时间写一下每个block的test吧，以防万一。

今天发现drop path也是模型架构敏感的，Baseline在加了一些drop path后效果下降了。

| arch         | RAF-DB | FERPlus | AffectNet |
| ------------ | ------ | ------- | --------- |
| Baseline     | 86.441 | 83.76   |           |
| CLS_only     | 87.256 | 82.944  |           |
| catBeforeMlp | 87.549 | -       |           |
| catAfterMlp  | 87.744 | 83.619  |           |
| addPatches   | 87.093 |         |           |

AffectNet先放置，测试一下把CLS expand加到patch上的效果

> Add patches效果确实不太行，还是直接cat吧。



# add with expand

在拼接patches之前，将cls token做expand，加到patches上。



# DropPath测试

这个测试确实没做，感觉不同大小的数据集对DropPath的需求不同，后面再查一查相关的信息。

嗯，droppath似乎也与epoch数相关。拉个长epoch试一试。



# 跑MultiScale

目前有两种划分MultiScale的想法：

1. 类似POSTER的思想，只取stage3，然后对stage3的channel上下采样。
2. 类似Unet的思想，取stage234。这个思路的话算量会比较大。

关键是怎么选择， 不确定有什么评判方法。
