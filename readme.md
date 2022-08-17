# 用deeplabv3+模型在Weizmann Horse进行实验

This repository uses the Weizmann Horse Database for training and semantic segmentation prediction of deeplabv3+ networks.

**[Encoder-decoder with atrous separable convolution for semantic image segmentation](https://arxiv.org/abs/1802.02611)**

## Installation
My code has been tested on Python 3.9 and PyTorch 1.11.0. Please follow the official instructions to configure your environment. See other required packages in `requirements.txt`.

## Model ##
模型采用deeplabv3+,主干网络部分采用轻量级的mobilenetv2
所有训练模型文件均存放在model文件夹下

````
model/
├──── bestmodel_***_***.pth   # 训练得到在测试集上最佳模型
│                             # ***部分从左往右依次为mIoU,boundary IoU
│                                
├──── mobilenet_v2.pth.tar    # 网络主干部分预训练模型,网址如下
│      # http://sceneparsing.csail.mit.edu/model/pretrained_resnet/mobilenet_v2.pth.tar
│ 
├──── freeze_model.pth        # 冻结主干训练25个epoch后整个网络模型
│                             
````

## Prepare Your Data

数据来源: [Weizmann Horse Database | Kaggle](https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database/metadata).
数据集包含327张马图片,及人工分割标注的ground truth
本次实验,随机取数据集85%作为训练集,剩下15%作为验证集与测试集

数据集在weizmann_horse_db下，文件结构如下

````
weizmann_horse_db/
├──── horse(327 images)   #大小不一的RGB图像
│    ├──── horse001.png
│    ├──── horse002.png
│    └──── ...
├──── mask(327images)     #0,1标注的gt图, 0表示背景1表示马
│    ├──── horse001.png
│    ├──── horse002.png
│    └──...
````


## Training

下载完成后,运行train从freeze.pth开始训练模型

    python train.py

- 测试集miou大于0.9模型保存到model下,训练过程中训练loss验证miou等参数随epoch变化曲线保存在log文件夹下
- 要想从头开始训练请更改train文件，去除freeze部分的注释
- 输入图片统一resize到224*224的大小输入网络
- 具体参数调节及本人更多详细训练方法参见train文件中注释
- 模型主干部分采用了公开的预训练模型，加快网络收敛
- 本实验在自己笔记本上用gpu完成, RTX2060-6G

## Testing ##

直接运行test.py,可以看到训练集与测试集对应miou与boundary iou, 以及随机两组图片的可视化预测文件保存在outputs文件夹下

    python test.py

- 详细说明见test文件中注释
- 最佳模型在测试集上**mIoU**指数为**0.904** , **Boundary IoU**指数为**0.757**。


## Permission and Disclaimer

This code is only for non-commercial purposes. the trained models included in this repository can only be used/distributed for non-commercial purposes. Anyone who violates this rule will be at his/her own risk.
