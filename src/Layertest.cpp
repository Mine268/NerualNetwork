#include "Layer.h"
#include <iostream>
using namespace std;


void test_forward_backward(){
    double x []={1,1,1};
    Sigmoid sigmoid;
    Layer layer(3,2,sigmoid);
    cout<<layer;
    layer.set_input(x);//输入
    layer.forward();//前向传播计算
    layer.read_output();//打印value
    layer.dubug();//打印integeration


    Layer layer1(2,1,sigmoid);
    cout<<layer1;
    layer1.set_input(layer.get_val());//输入
    layer1.forward(false);//前向传播计算,到输出层不需要使用激活函数
    layer1.read_output();//打印value
    layer1.dubug();//打印integeration

    //反向传播测试
    double ddelta[] = {-0.22};
    layer1.set_ddelta(ddelta);
    layer1.debug2();
//
    layer.backward(layer1);
    layer.debug2();
}
void test_copy_operator(){
    Sigmoid sigmoid;
    Layer first(2,4,sigmoid);
    Layer two(first);
    cout<<first;
    cout<<two;

}

int main() {
//    test_forward_backward();
    test_copy_operator();

}
