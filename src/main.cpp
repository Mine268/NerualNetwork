#include <iostream>
#include "../lib/Network.h"

int main() {
	Network nn({Network::layer(3, func_ReLU),
				Network::layer(2, func_ReLU),
				Network::layer(1, func_ReLU)});

    nn.print_network(std::cout);

    auto res = nn.evaluate(new double[3]{1., 1.,1.0});

    for (std::size_t i = 0; i < 1; ++i)
        std::cout << res[i] << ' ';

//    nn.single_fit(1.0,new node_type[1]{1.0});
//
//    cout<<endl;
//    nn.print_network(std::cout);

	return 0;
}
