//
// Created by mine268 on 2021/7/13.
// IFunction接口，用于表示激活函数
//

#ifndef NERUALNETWORK_IFUNCTION_H
#define NERUALNETWORK_IFUNCTION_H

#include "./cxr_header.h"

class IFunction {
public:
    virtual node_type activation(node_type) = 0;
    virtual node_type d_activation(node_type) = 0;

};

class Sigmoid : public virtual IFunction {
public:
    virtual node_type activation(node_type value) {
        return 1 / (1 + exp(-value));
    }

    virtual node_type d_activation(node_type value) {
        double sigma = activation(value);
        return sigma * (1 - sigma);
    }
};

class Tanh : public virtual IFunction {
public:
    virtual node_type activation(node_type value) {
        return tanh(value);
    }

    virtual node_type d_activation(node_type value) {
        return 1 - pow(activation(value), 2.);
    }
};

class ReLU : public virtual IFunction {
public:
    virtual node_type activation(node_type value) {
        return fmax(value, 0.);
    }

    virtual node_type d_activation(node_type value) {
        return value > 0 ? 1 : 0;
    }
};

extern Sigmoid func_sigmoid;
extern Tanh func_tanh;
extern ReLU func_ReLU;

#endif //NERUALNETWORK_IFUNCTION_H
