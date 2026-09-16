---
title: python深度学习库熟练
top: false
cover: false
toc: true
mathjax: true
date: 2025-12-24 22:05:50
password:
summary:
tags:
categories:
---

以记英文为主，记中文没有用（对于我自己）

# day1 基础语法熟练

Python PEP8 编码规范: 先跟着小项目记录一部分
```python
# 1. 类名首字母大写
```

## 基本数据结构

data type ：数据类型，所以 `type(a)` 来获得 data type.

- Python是动态类型语言，变量的类型根据情况自动决定的

列表部分：

- 主要是slicing标记


## 相关部分：

### Python的类继承

1. 抽象方法必须改写，声明方式`@abstractmethod`

2. 初始化参数不同时，需要调用父类的初始化`super().__init__(args_father)`

3. 抽象类如果想强制必须检查函数有无实现抽象否则报错还可以，引入ABC和abstractmethod

```python
from abc import ABC, abstractmethod

class father(ABC):
    @abstractmethod
    def func(self, args):
        pass

class son(father):
    def func(self, args):
        # 具体实现
```

### Numpy的多维数组 ndarray

1. 排列方向成为轴或者维

    - ndarray实例中有一个实例变量叫 ndim（number of dimensions），表示多维数组的维度/轴数。

    - 维度这一个词，在那种多维方阵下是指轴数，但在向量的情景下是指个数，比如n维向量是n个。