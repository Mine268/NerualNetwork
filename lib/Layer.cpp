//
// Created by God on 2021/7/13.
//

#include "Layer.h"
#include <random>

Layer::Layer(int _size, const Layer *pre, action function) {
    select_action = function;
    set_function();



    size = _size;
    if (pre != nullptr) {
        //如果不是第一层,则需要初始化weight矩阵和bias
        pre_size = pre->getsize();
        weights = new node_type *[pre_size];
        for (int i = 0; i < pre_size; i++) weights[i] = new node_type[size];

    } else {
        pre_size = 0;
    }
}
Layer::Layer(int _size, double input[]) {
    //初始化输入层
    size =_size;pre_size = 0;
    value = new node_type[size];
    for (int i=0;i<size;i++) value[i] = input[i];
}


int Layer::getsize() const {
    return size;
}

void Layer::set_function() {
    switch (select_action) {
        case 1: actfunction = new class Sigmoid;
        case 2: actfunction = new class ReLU;
        case 3: actfunction = new class Tanh;
    }
}

Layer::~Layer() {


}


void Layer::read_output() const {

}

void Layer::forward_propagation(const Layer*pre) {

}

void Layer::backward_propagation() {

}
