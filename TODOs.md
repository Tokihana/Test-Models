# IRback

因为输入是更大的224x224图像，所以首先将conv1改为stride2，测试一下效果。

如果效果损失比较厉害的话，则fc_scale用14*14。



## 选择合适的batchsize

通常情况下，更换架构就意味着更换batchsize。

以2的幂次为步进，测试硬件最高能够承受的batchsize范围

测试一下throughput：

On RTX4060Ti

| batch size | throughput<br />(example per second) | time per step<br />(time per batch) |
| ---------- | ------------------------------------ | ----------------------------------- |
| 1          | 92                                   | 0.01                                |
| 2          | 145                                  | 0.01                                |
| 4          | 164                                  | 0.02                                |
| 8          | 150                                  | 0.05                                |
| 16         | 161                                  | 0.10                                |
| 32         | 156                                  | 0.20                                |
| 64         | 157                                  | 0.41                                |



