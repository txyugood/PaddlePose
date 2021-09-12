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
nohup python -u train.py --image_size 256 > hourglass_256.log &
tail -f hourglass_256.log 
```

训练图像尺寸为384的模型。

```
nohup python -u train.py --image_size 384 > hourglass_384.log &
tail -f hourglass_384.log 
```
以上在后台训练模型，并输出日志到hourglass_xxx.log文件，通过tail命令实时查看训练日志。

# 5.验证模型

预训练模型下载地址:

下载模型后使用，下列命令验证模型：
```
python val.py --image_size 256 --pretrained_model 
```
验证结果：
```
[EVAL] Ankle=79.87761299600484 Elbow=89.09163062349077 Head=96.65757162346522 Hip=88.41959160211289 Knee=83.8608487080676 Mean=88.71714806141036 Mean@0.1=32.10772823107419 Shoulder=95.36345108695652 Wrist=83.77702302257738 
```

```
python val.py --image_size 384 --pretrained_model 
```

# 5总结
以下表格是本次论文复现的结果。
| Arch  | Input Size | Mean@0.1 | pytorch Mean@0.1 |
| :--- | :--------: | :------: | :------: | 
| pose_hourglass_52 | 256x256 | 0.321 | 0.317
| pose_hourglass_52 | 384x384 | 0.321 | 0.366

本次论文复现是我第一次接触人体姿态估计这个领域，通过复现过程学习了很多知识，感谢百度飞桨提供这次比赛，让我学习到了很多。