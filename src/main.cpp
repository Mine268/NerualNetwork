#include <iostream>

#include "../lib/Network.h"

int main() {
	Network nn({Network::layer(3, func_sigmoid), Network::layer(2, func_sigmoid),
				Network::layer(1, func_sigmoid)});

	nn.print_network(std::cout);

	auto res = nn.evaluate(new double[3]{1., 1., 1.0});

	for (std::size_t i = 0; i < 1; ++i) cout << res[i] << ' ';

	for (std::size_t i = 0; i < 1000; ++i) {
		nn.single_fit(.1, new node_type[1]{0.});
		nn.evaluate(new double[3]{1., 1., 1.0});
	}

	cout << endl;
    res = nn.evaluate(new double[3]{1., 1., 1.0});
	for (std::size_t i = 0; i < 1; ++i) cout << res[i] << ' ';

    cout << endl << "-------" << endl;
    nn.print_network(std::cout);

	return 0;
}
