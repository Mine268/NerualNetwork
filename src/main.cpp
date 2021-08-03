#include <iostream>
#include "../lib/Network.h"

int main() {
	Network nn({Network::layer((std::size_t)10, func_sigmoid),
				Network::layer((std::size_t)12, func_sigmoid),
				Network::layer((std::size_t)10, func_sigmoid)});
	return 0;
}
