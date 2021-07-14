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
    int input_size;
    int output_size;
    valarray<node_type >biases;
    valarray<node_type >val;
    valarray<node_type >integration;
    node_type **weight;
    valarray<node_type >ddelta;
    Layer*nextLayer;
    Layer*preLayer;
    IFunction*activation;
public:
    enum action{Sigmoid=1,ReLU,Tanh};
    Layer(int _inputsize,int _outputsize,action function);
    ~Layer();
    void set_input(node_type*input);
    void read_output()const;
    void forward();
    void backward();
    void get_val();
};
#endif //NERUALNETWORK_LAYER_H
