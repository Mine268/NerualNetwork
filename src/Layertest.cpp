#include "Layer.h"
#include <iostream>
#include "windows.h"
using namespace std;


void test_forward_backward(){
    double x []={1,1,1};
    ReLU reLu;
    Layer layer(3,2,reLu);
    cout<<layer;
    layer.set_input(x);//输入
    layer.forward_propagation();//前向传播计算
    layer.read_output();//打印value
    layer.debug();//打印integeration
    cout<<endl;

    Layer layer1(2,1,reLu);
    cout<<layer1;
    layer1.set_input(layer.get_val());//输入
    layer1.forward_propagation(true);//前向传播计算,到输出层使用激活函数
    layer1.read_output();//打印value
    layer1.debug();//打印integeration

    cout<<endl;
    //反向传播测试
    //假设目标值是1
    double ddelta[] = {-(1.0-1.76964)*1};
    layer1.set_ddelta(ddelta);
    layer1.back_propagation(nullptr,1);
    layer1.debug2();//打印delta

    cout<<"反向传播后："<<endl;
    cout<<layer1;
    cout<<endl;
    layer.back_propagation(&layer1,1);
    layer.debug2();


    cout<<"反向传播后："<<endl;
    cout<<layer;
}
void test_save_load(){
    Sigmoid sigmoid;
    Layer first(4,5,sigmoid);
    cout<<first;
    first.save_data("../data");
    Sleep(1000);
    Layer test(4,5,sigmoid);
    cout<<test;
    test.load_data("../data");
    cout<<test;

}

int main() {
//    test_forward_backward();
    test_save_load();

}
