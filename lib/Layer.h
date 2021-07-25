//
// Created by God on 2021/7/13.
//

#ifndef NERUALNETWORK_LAYER_H
#define NERUALNETWORK_LAYER_H

#include <valarray>
#include "cxr_header.h"
#include <iostream>

using namespace std;

class Layer {
private:
    int input_size;
    int output_size;
    node_type *biases;
    node_type *value; //a = f(z)
    node_type *integeration;//z =w x+b
    node_type **weight;
    node_type *preval;//前一层的激活值
    node_type *ddelta;//
    node_type *dbiases;//最终版本应该会删除，此处只用于测试层是否工作正常
    node_type **dweight;//也会被删除



    IFunction &activation;

    // 将全体数据随机化生成，随机化依据是标准正态分布
    void init_wb();
    // 初始化全体数据内存，内部数据未定义
    void new_data();

public:
    // 构造
    // 函数，参数含义分别是
    // - 本层的节点个数
    // - 下一层的节点个数
    // - 激活函数
    Layer(int _inputsize, int _outputsize, IFunction& function);

    //复制构造函数
    Layer(const Layer&);

    //销毁申请的空间
    ~Layer();

    // 设置本层的输入数据，即上一层的节点数据
    void set_input(const node_type *input);

    //返回激活值
    const node_type *get_val() ;

    //打印当前层的激活值
    void read_output() const;

    //前向传播 根据上一层的数据计算这一层的数据
    void forward_propagation(bool use_activation = true);

    //反向传播 根据next层的数据计算这一层的反向传播并减到对应参数上
    void back_propagation(Layer &preLayer,node_type learning_rate);

    //获取激活函数
    IFunction& get_IFunction();

    //传入下一层的ddelta
    void set_ddelta(node_type *delta);

    //打印层的相关信息
    friend ostream &operator<<(ostream &os, Layer &one);

    void debug();

    void debug2();
};

#endif //NERUALNETWORK_LAYER_H
