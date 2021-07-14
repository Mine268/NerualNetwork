//
// Created by God on 2021/7/13.
//

#include "Layer.h"
#include <random>
#include <iterator>
#include <ctime>

Layer::Layer(int _inputsize, int _outputsize, action function) {
    input_size = _inputsize;
    output_size = _outputsize;
    weight = new node_type *[input_size];
    for (int i=0;i<input_size;i++) {
        weight[i] = new node_type [output_size];
    }
    biases = new node_type [output_size];
    value = new node_type [output_size];
    integration = new node_type [output_size];
    ddelta = new node_type [output_size];
    inputdata = new node_type [input_size];
    set_function(function);
    inin_wb();
}
Layer::~Layer() {
    for (int i=0;i<input_size;i++){
        delete []weight[i];
    }
    delete actfunction;
    delete []biases;
    delete []value;
    delete []integration;
    delete []ddelta;
    delete [] inputdata;
}
void Layer::set_function(int choose) {
    switch (choose) {
        case 1:
            actfunction = new class Sigmoid;
            break;
        case 2:
            actfunction = new class ReLU;
            break;
        case 3:
            actfunction = new class Tanh;
            break;
    }
}

void Layer::inin_wb() {
    auto seed = (unsigned)time(0);
    random_device device;
    default_random_engine engine_need(seed);
    normal_distribution<node_type >distribution;
    for(int i=0;i<input_size;i++){
        for (int j=0;j<output_size;j++){
            weight[i][j] = distribution(engine_need);
        }
    }
    for(int i=0;i<output_size;i++){
        biases[i] = 0;
    }
}

node_type *Layer::get_val() {
    return value;
}

void Layer::set_input(node_type *input) {
    copy(input,input+input_size,inputdata);
}

void Layer::read_output() const {
    for_each(value,value+output_size,[](node_type val){cout<<val<<" ";});
    cout<<endl;
}

ostream &operator<<(ostream& os,Layer&one){
    os<<"input_size: " <<one.input_size<<endl;
    os<<"output_size: "<<one.output_size<<endl;
    os<<"weight: "<<endl;
    for(int i=0;i<one.input_size;i++){
        for(int j=0;j<one.output_size;j++){
            cout<<one.weight[i][j]<<" ";
        }
        cout<<endl;
    }
    os<<"biases: "<<endl;
    for_each(one.biases,one.biases+one.output_size,[](node_type val){cout<<val<<" ";});
    cout<<endl;
    return os;
}

void Layer::forward() {
    for(int i=0;i<output_size;i++){
        node_type z=biases[i];
        for(int j=0;j<input_size;j++){
            z += inputdata[j]*weight[j][i];

        }
        integration[i] = z;
        value[i] = actfunction->activation(z);
    }
}

void Layer::backward() {

}