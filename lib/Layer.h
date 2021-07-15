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
    node_type *value; //a = f(z)
    node_type *integration;//z =w x+b
    node_type **weight;
    node_type *ddelta;//
    node_type *dbiases;
    node_type  **dweight;
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
    void set_input(node_type*input);//传入上一层的value
    void read_output()const;
    void forward(bool useactfunction= true);
    void backward(Layer &next);
    node_type *get_val();
    void set_ddelta(node_type *delta);  //传入下一层的ddelta
    friend ostream &operator<<(ostream& os,Layer&one);
    void dubug();
    void debug2();
};
#endif //NERUALNETWORK_LAYER_H
