//
// Created by mine268 on 2021-8-21
// 这个类用于神经网络构造时传递层信息使用
//

#ifndef NERUALNETWORK_LAYERCONSTRUCTIONINFO_H
#define NERUALNETWORK_LAYERCONSTRUCTIONINFO_H

#include <iostream>
#include "cxr_header.h"

struct LayerConstructionInfo {
    std::size_t size;
    IFunction * activation;
};

#endif