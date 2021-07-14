#include "Layer.h"
#include <iostream>
#include <valarray>
#include "../lib/cxr_header.h"
using namespace std;



int main() {
    double x []={1,1,1,1};
    valarray<double>xx(x, sizeof(x)/sizeof(x[0]));
    cout<<xx.size()<<endl;
    for_each(x,x+4,[](double val){cout<<val<<" ";});
    cout<<endl;
    double *y = new double [4];
    copy(x,x+4,y);
    for_each(y,y+4,[](double val){cout<<val<<" ";});
    cout<<endl;
    Layer layer(4,3,Layer::ReLU);

    cout<<layer;
    layer.set_input(x);
    layer.forward();
    layer.read_output();
}
