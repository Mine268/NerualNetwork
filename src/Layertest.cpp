#include "Layer.h"
#include <iostream>
#include <valarray>
#include <vector>
using namespace std;


void Test(){
    double x[] = {1.0,2,4};
     Layer *s =new Layer(3, x);
    Layer (10,s,Layer::Sigmoid);
    cout<<"yes\n";
}

int main() {
    Test();

}
