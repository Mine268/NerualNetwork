# README

## 简介

这是一个简单的神经网络库，支持基础的全连接前馈神经网络的构造、训练、*存储*。本程序基于`C++`开发，仅供学习交流使用，切勿投入实际运用。

## 使用方法

1. 导入头文件

   ```cpp
   #include "cxr_nn.h"
   ```

2. 建立相应的神经网络

   ```cpp
   // 建立神经网络nw
   Network nw(Network.Layer(28*28, func_sigmoid),
              Network.Layer(13, func_sigmoid),
              Network.Layer(10, func_sigmoid),
              Network.Layer(10, func_sigmoid));
   ```

   这个网络输入是一个`28*28`的向量，输出一个10维的向量，层与层之间完全连接，使用`sigmoid`作为激活函数，可以使用的激活函数默认包括三种

   - `func_sigmoid` 经典的sigmoid函数
   - `func_tanh` 双曲正切函数
   - `func_ReLU` ReLU函数

   > 也可以由使用者自己来写激活函数，继承接口即可。

3. 使用idx1/idx3格式的文件进行训练

   ```cpp
   nw.fit(1., 30, 50, "./mnist/train-image-idx3-ubyte", "./mnist/train-label-idx1-ubyte");
   ```

   使用idx1/idx3格式的文件进行训练，指定学习率为1，训练轮数为30，每次训练的样本数量为50(batch-size)，训练的数据来自

   - `train-image-idx3-ubyte` 训练的数据文件
   - `train-label-idx1-ubyte` 训练的标签文件

   > 可以加入每一次epoch后检查训练的正确率的功能。
   >
   > ```cpp
   > nw.fit(1., 30, 50, "./mnist/train-image-idx3-ubyte", "./mnist/train-label-idx1-ubyte", "./mnist/test-image-idx3-ubyte", "./mnist/tes-label-idx1-ubyte");
   > ```
   >
   > - `test-image-idx3-ubyte` 验证的数据文件
   > - `test-label-idx1-ubyte` 验证的标签文件

   训练的方式为随机梯度下降，采用平均平方函数作为代价函数。

   > 或许以后可以改成提供若干种代价函数以供选择。

4. 使用神经网络

   ```cpp
   double result[] = nw.evaluate(new double[]{0.49, 0.03, ..., 0.00});
   ```


## -

就三个大二计科练手用的，没有什么技术含量，甚至不是人工智能专业的。

[@Godones](https://github.com/Godones)

[@Mine268](https://github.com/Mine268)

[@xzhy324](https://github.com/xzhy324)

