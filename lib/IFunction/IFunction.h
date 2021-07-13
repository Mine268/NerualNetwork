//
// Created by mine268 on 2021/7/13.
// IFunction接口，用于表示激活函数
//

#ifndef NERUALNETWORK_IFUNCTION_H
#define NERUALNETWORK_IFUNCTION_H

#include "../cxr_header.h"

class IFunction {
public:
    virtual node_type activation(node_type);
    virtual node_type d_activation(node_type);
};


#endif //NERUALNETWORK_IFUNCTION_H
