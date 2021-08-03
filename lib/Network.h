//
// Created by mine268 on 2021-8-2
//

#ifndef NERUALNWTWORK_NETWORK_H
#define NERUALNWTWORK_NETWORK_H

#include <iostream>

#include "cxr_header.h"
#include "IFunction.h"
#include "Layer.h"
#include "LayerConstructionInfo.h"

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
	// 构造函数，用于构造network类，接受一个初始化列表，列表中的每一个元素是指向LayerConstrunctionInfo类的指针，这个类中存储了构造层所必要的信息，通过使用本类的静态方法来获得这样的一个指针。LayerConstructionInfo类中存储了层的大小，和这一层应用的激活函数（IFunctino）。
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
	static LayerConstructionInfo* layer(std::size_t size, IFunction& func);
};

#endif
