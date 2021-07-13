#include <iostream>
#include "../lib/cxr_header.h"
#include "../lib/IFunction/IFunction.h"

using namespace std;

int main() {
    IFunction & ifc = func_sigmoid;

    cout << ifc.activation(7.2) << endl;

    return 0;
}