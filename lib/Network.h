//
// Created by mine268 on 2021-8-2
//

#ifndef NERUALNWTWORK_NETWORK_H
#define NERUALNWTWORK_NETWORK_H

#include <iostream>

#include "Layer.h"
#include "LayerConstructionInfo.h"
#include "cxr_header.h"

class Network {
   private:
	// 神经网络的层
	Layer** layers;
	// 神经网络的层数
	std::size_t layer_count;
    // 神经网络的每层的大小
    std::size_t* layer_length;

   public:
	// 通过初始化列表的方式实现不定参数
	Network(std::initializer_list<LayerConstructionInfo*>);
	~Network();

	// 进行训练
    // - learningRate 训练的学习率
    // - epoch 训练轮数
    // - batch_size 每次取的训练样本的大小
    // - data_path 训练数据输入的文件
    // - label_path 训练标签的位置
	void fit(double learningRate, int epoch, int batch_size,
			 std::string data_path, std::string label_path);
    // 依据输入进行计算得到输出
    const double* evaluate(node_type data[]);

    // 这是在构造时调用的静态函数
    static LayerConstructionInfo * layer(std::size_t size, IFunction & func);
};

#endif
