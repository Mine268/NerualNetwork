#include <iostream>
#include "../lib/Network.h"

int main() {
	Network nn({Network::layer(2, func_sigmoid),
				Network::layer(3, func_sigmoid),
				Network::layer(3, func_sigmoid)});

    nn.print_network(std::cout);

    auto res = nn.evaluate(new double[2]{1., -1.});

    for (std::size_t i = 0; i < 3; ++i)
        std::cout << res[i] << ' ';

    std::cout << std::endl;

	return 0;
}
