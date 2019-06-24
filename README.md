## 介绍
1. 使用flask+keras构建一个云朵识别API 对云朵图片进行分类
2. 模型为cloudNet以及迁移学习的RestNet进行对比
## 使用方式
1. 启动predict.py：开启Flask服务器，调用模型，准备接收数据
2. 在页面中传入图片，之后点击判断，获得所属类别
## 数据来源
1. 感谢南京信息工程大学大气学院张教授团队提出的云分类卷积网络CloudNet以及Cirrus Cumulus Stratus Nimbus(CCSN)的地面云图数据集
2. 数据集中包含11类云朵
3. 论文标题：*Zhang, J. L., Liu, P., Zhang, F., & Song, Q. Q. (2018). CloudNet: Ground-based Cloud Classification with Deep Convolutional Neural Network. Geophysical Research Letters, 45.*
4. [论文地址](https://doi.org/10.1029/2018GL077787)