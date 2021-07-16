//
// Created by mine268 on 2021/7/13.
// 一个常数、类型的头文件
//

#ifndef NERUALNETWORK_CXR_HEADER_H
#define NERUALNETWORK_CXR_HEADER_H

#include <cmath>

typedef double node_type;


class IFunction {
public:
    virtual node_type activation(node_type)=0;
    virtual node_type d_activation(node_type)=0;
};

class Sigmoid : public virtual IFunction {
public:
    virtual node_type activation(node_type value) {
        return 1 / (1 + exp(value));
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
        return fmax(value, 0);
    }

    virtual node_type d_activation(node_type value) {
        return value > 0 ? 1 : 0;
    }
};

#endif //NERUALNETWORK_CXR_HEADER_H