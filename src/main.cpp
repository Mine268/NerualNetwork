#include <iostream>
#include "../lib/Network.h"

int main() {
	Network nn({Network::layer(10, func_sigmoid),
				Network::layer(12, func_sigmoid),
				Network::layer(10, func_sigmoid)});
	return 0;
}
