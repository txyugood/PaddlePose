# 1.简介
本项目基于PaddlePaddle复现《Stacked Hourglass Networks for Human Pose Estimation》论文，该论文提出了一种人体姿态估计的方法，在MPII数据集上达到如下精度：

 size:384x384, mean@0.1: 0.366
size:256x256, mean@0.1: 0.317

# 2.数据集下载

MPII:[https://aistudio.baidu.com/aistudio/datasetdetail/107551](https://aistudio.baidu.com/aistudio/datasetdetail/107551)

# 3.环境

PaddlePaddle == 2.1.2

python == 3.7

# 4. 训练

训练图像尺寸为256的模型。

```
nohup python -u train.py > hourglass_256.log &
tail -f hourglass_256.log 
```

训练图像尺寸为384的模型。

```
nohup python -u train.py > hourglass_384.log &
tail -f hourglass_384.log 
```
以上在后台训练模型，并输出日志到hourglass_xxx.log文件，通过tail命令实时查看训练日志。

5.验证模型

预训练模型下载地址:

下载模型后使用，下列命令验证模型：


# 5总结

