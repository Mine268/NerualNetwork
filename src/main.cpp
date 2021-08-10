#include <iostream>

#include "../lib/FileReader.h"
#include "../lib/Network.h"

int main() {
	Network nn(
		{Network::layer(784, func_sigmoid), Network::layer(50, func_sigmoid),
		 Network::layer(20, func_sigmoid), Network::layer(10, func_sigmoid)});
	FileReader fr(10, "D:\\study\\CLionProjects\\net\\NerualNetwork\\train\\t10k-images.idx3-ubyte",
                  "D:\\study\\CLionProjects\\net\\NerualNetwork\\train\\t10k-labels.idx1-ubyte");

	node_type *d_data = new node_type[784];
	const node_type *d_result;
	data_type *i_data, *i_result;

	nn.fit(0.35, 1, 1, "D:\\study\\CLionProjects\\net\\NerualNetwork\\train\\train-images.idx3-ubyte",
           "D:\\study\\CLionProjects\\net\\NerualNetwork\\train\\train-labels.idx1-ubyte");

	int count = 0;
	for (int k = 0; k < 5000; k++) {
		i_data = fr.getData();
		i_result = fr.getLabel();
		for (std::size_t i = 0; i < 784; ++i) d_data[i] = (node_type)i_data[i];
		d_result = nn.evaluate(d_data);
		
		double max1 = -1.;
		int max2 = -1.;
		int maxi1 = 0, maxi2 = 0;
		for (int i = 0; i < 10; ++i) if (d_result[i] > max1) max1 = d_result[i], maxi1 = i;
		for (int i = 0; i < 10; ++i) if (i_result[i] > max2) max2 = i_data[i], maxi2 = i;

		if (maxi1 == maxi2) ++count;
	}
	std::cout << (double)count / 5000;

	return 0;
}
