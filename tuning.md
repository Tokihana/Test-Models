# IRback

因为输入是更大的224x224图像，所以首先将conv1改为stride2，测试一下效果。

如果效果损失比较厉害的话，则fc_scale用14*14。



## batch size

通常情况下，更换架构就意味着更换batchsize，所以每个架构都需要做一次batchsize测试。

### 最大可承受batchsize

以2的幂次为步进，测试硬件最高能够承受的batchsize范围

### throughput

> 做throughput的目的是检查是否存在训练流程上的瓶颈，例如I/O或者多卡并行的同步点。

$$
throughput = (*examples\ per\ second) = \frac {(*num\ of\ examples)} {(*time)}\\
time\ per\ step(batch) = \frac {batch\ size}{throughput}
$$

On RTX4060Ti（**IR50**）

| batch size | throughput<br />(example per second) | time per step<br />(time per batch) |
| ---------- | ------------------------------------ | ----------------------------------- |
| 1          | 92                                   | 0.01                                |
| 2          | 145                                  | 0.01                                |
| 4          | 164                                  | 0.02                                |
| 8          | 150                                  | 0.05                                |
| 16         | 161                                  | 0.10                                |
| 32         | 156                                  | 0.20                                |
| 64         | 157                                  | 0.41                                |

On RTX3090（**IR50**)

| batch size | throughput<br />(example per second) | time per step<br />(time per batch) |
| ---------- | ------------------------------------ | ----------------------------------- |
| 1          | 151                                  | 0.007                               |
| 2          | 318                                  | 0.006                               |
| 4          | 346                                  | 0.012                               |
| 8          | 364                                  | 0.022                               |
| 16         | 414                                  | 0.039                               |
| 32         | 447                                  | 0.071                               |
| 64         | 450                                  | 0.142                               |
| 128        | 458                                  | 0.279                               |
| 256        | 465                                  | 0.550                               |
| 512        | 464                                  | 1.101                               |
| 1024       | 显存满了                             |                                     |

很有意思的现象，显存虽然没满，但throughput明显偏向饱和了。

且time per step在很小的batch之后，就开始翻倍，大概是开始排队了。

这应该也就代表，hardware saturated不代表显存满了，只是核全被用上了。显存可能更多与图像的大小有关。

尝试将图像大小压小一些（224 -> 112），看看是不是可以跑1024的batch size

| batch size | throughput | time per step |
| ---------- | ---------- | ------------- |
| 1          | 161        | 0.006         |
| 2          | 310        | 0.006         |
| 4          | 610        | 0.007         |
| 8          | 1231       | 0.006         |
| 16         | 1471       | 0.011         |
| 32         | 1546       | 0.021         |
| 64         | 1662       | 0.038         |
| 128        | 1777       | 0.072         |
| 256        | 1812       | 0.141         |
| 512        | 1826       | 0.280         |
| 1024       | 1835       | 0.558         |

确实可以跑，这样一看，如果图像尺寸不是大的非常离谱，显存通常都是有富余的。

总结一下：

- 显存占用量通常与data scale的关联更大，更大的图像、更大的中间张量对显存的占用更加明显
- throughput对应着硬件本身的计算速度，在运算单元全部被占用（saturated）之前，增加batch_size，等价于增加还未被使用的运算单元，此时batch_size加倍，对应throughput加倍，time_per_step不变。这个效应通常会在batch_size增加到一定程度后饱和，变为batch_size加倍、time_per_step加倍（开始排队），throughput可能会继续增加，但逐渐接近某个上限。
- 如果throughput已经到达了增长的上限，此时继续加大batchsize并不会带来运行效率上的提升，就没必要再加了

### 通过throughput评估训练时间，选择最快的batchsize

$$
training\ time = (*time\ per\ step) * (*num\ steps)
$$

