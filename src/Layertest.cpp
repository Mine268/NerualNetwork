#include "Layer.h"
#include <iostream>
using namespace std;


void test_forward_backward(){
    double x []={1,1,1};
    ReLU reLu;
    Layer layer(3,2,reLu);
    cout<<layer;
    layer.set_input(x);//����
    layer.forward_propagation();//ǰ�򴫲�����
    layer.read_output();//��ӡvalue
    layer.debug();//��ӡintegeration
    cout<<endl;

    Layer layer1(2,1,reLu);
    cout<<layer1;
    layer1.set_input(layer.get_val());//����
    layer1.forward_propagation(true);//ǰ�򴫲�����,�������ʹ�ü����
    layer1.read_output();//��ӡvalue
    layer1.debug();//��ӡintegeration

    cout<<endl;
    //���򴫲�����
    //����Ŀ��ֵ��1
    double ddelta[] = {-(1.0-1.76964)*1};
    layer1.set_ddelta(ddelta);
    layer1.back_propagation(nullptr,1);
    layer1.debug2();//��ӡdelta

    cout<<"���򴫲���"<<endl;
    cout<<layer1;
    cout<<endl;
    layer.back_propagation(&layer1,1);
    layer.debug2();



    cout<<"���򴫲���"<<endl;
    cout<<layer;
}
void test_copy_operator(){
    Sigmoid sigmoid;
    Layer first(2,4,sigmoid);
    Layer two(first);
    cout<<first;
    cout<<two;

}

int main() {
    test_forward_backward();
//    test_copy_operator();

}
