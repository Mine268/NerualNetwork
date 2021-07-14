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
    node_type *biases;
    node_type *value;
    node_type *integration;
    node_type **weight;
    node_type *ddelta;

    Layer*nextLayer;
    Layer*preLayer;
    IFunction *actfunction;

    void set_function(int a);
    void inin_wb();
    node_type *inputdata;

public:
    enum action{Sigmoid=1,ReLU,Tanh};
    Layer(int _inputsize,int _outputsize,action function = Layer::Sigmoid);
    ~Layer();
    void set_input(node_type*input);
    void read_output()const;
    void forward();
    void backward();
    node_type *get_val();
    friend ostream &operator<<(ostream& os,Layer&one);
    void dubug();
};
#endif //NERUALNETWORK_LAYER_H
