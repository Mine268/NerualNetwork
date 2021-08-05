#include <iostream>
#include "../lib/Network.h"

int main() {
	Network nn({Network::layer(2, func_sigmoid),
				Network::layer(3, func_sigmoid),
				Network::layer(3, func_sigmoid)});

    nn.print_network(std::cout);

    auto res = nn.evaluate(new double[2]{1., -1.});
    auto target = new double[3]{1., 0., .5};

    // 打印一下没有训练的结果
    for (std::size_t i = 0; i < 3; ++i)
        std::cout << res[i] << ' ';

    for (std::size_t i = 0; i < 25565; ++i) {
        nn.single_fit(1., target);
        nn.evaluate(new double[2]{1., -1.});
    }

    res = nn.evaluate(new double[2]{1., -1.});
    std::cout << std::endl;
    for (std::size_t i = 0; i < 3; ++i)
        std::cout << res[i] << ' ';

    std::cout << std::endl;

	return 0;
}
