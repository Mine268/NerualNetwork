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

    node_type *ddelta;//
    node_type *dbiases;//最终版本应该会删除，此处只用于测试层是否工作正常
    node_type **dweight;//也会被删除
    node_type *preval;//前一层的激活值


    IFunction &activation;
    void init_wb();//初始化weight与biases
    void new_data();//申请空间

public:

    Layer(int _inputsize, int _outputsize, IFunction& function);//初始化层

    Layer(const Layer&);//复制构造函数

    ~Layer();//销毁申请的空间

    void set_input(const node_type *input);//传入数据

    const node_type *read_output() const;//返回激活值

    void forward_propagation(bool use_activation = true);//前向传播

    void back_propagation(Layer &preLayer);//反向传播

    IFunction& get_IFunction();//获取激活函数

    void set_ddelta(node_type *delta);  //传入下一层的ddelta

    friend ostream &operator<<(ostream &os, Layer &one); //打印层的相关信息

    void debug();

    void debug2();
};

#endif //NERUALNETWORK_LAYER_H
