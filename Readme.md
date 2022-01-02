# README

## 环境配置

推荐使用conda管理环境，但我们同样导出了requirements.txt， Jittor的安装建议查找[官网](https://cg.cs.tsinghua.edu.cn/jittor/download/)。

```sh
conda env create -f freeze.yml
```

```sh
pip install -r requirements.txt
```

## 结构

torch文件夹是 Pytoch 实现，crnn_jittor 文件夹是Jittor实现，demo文件夹是LibTorch的demo，tools里面主要是建立训练集/测试集的代码。

## 训练

```sh
python main.py --version 1
```
