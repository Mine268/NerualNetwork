#include "Layer.h"
#include <iostream>
#include <valarray>
#include "../lib/cxr_header.h"
using namespace std;



int main() {
    double x []={1,1,1};
    Layer layer(3,2,Layer::ReLU);
    cout<<layer;
    layer.set_input(x);//输入
    layer.forward();//前向传播计算
    layer.read_output();//打印value
    layer.dubug();//打印integeration


    Layer layer1(2,1,Layer::ReLU);
    cout<<layer1;
    layer1.set_input(layer.get_val());//输入
    layer1.forward(false);//前向传播计算,到输出层不需要使用激活函数
    layer1.read_output();//打印value
    layer1.dubug();//打印integeration

    //反向传播测试
    //假设误差函数使用均方误差，目标值为0
    double ddelta[] = {-0.113};//需要计算并手动设置最后一层的ddelta
    layer1.set_ddelta(ddelta);

    layer1.backward();
    layer1.debug2();
    layer.backward(&layer1);//前面的层只需要将后一层传入即可
    layer.debug2();
}
