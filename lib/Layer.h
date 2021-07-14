//
// Created by God on 2021/7/13.
//

#ifndef NERUALNETWORK_LAYER_H
#define NERUALNETWORK_LAYER_H

#include <valarray>
#include "cxr_header.h"
#include <iostream>
using namespace std;

class Layer{
private:

    int size;                           //当前层的大小
    int pre_size;                       //前一层的大小
    Layer *preLayer;
    Layer *nextLayer;
    IFunction *actfunction;
    node_type *value;                   // 激活值f(z)
    node_type *integeration;            //z = wx+b
    node_type *biases;                  //b
    node_type **weights;                //前一层到这一层的w;
    node_type *ddelta;
    int select_action;
    void set_function();
public:
    enum action{Sigmoid=1,ReLU,Tanh};
    Layer(int _size,const Layer *pre, action function);
    int getsize()const;
    ~Layer();
    explicit Layer(int _size,node_type input[]);
    void read_output()const;
    void forward_propagation(const Layer *pre);//根据前一层的值计算这一层的值
    void backward_propagation();
    friend ostream& operator <<(ostream&,const Layer*layer);//输出层相关的信息


};
#endif //NERUALNETWORK_LAYER_H
