#include "Network.h"

Network::Network(std::initializer_list<LayerConstructionInfo*> list)
	: layers(nullptr), layer_count(0), layer_length(nullptr) {
	// 首先计算出总共需要多少层
	for (auto lyrInfo : list) ++layer_count;

	// 分配空间
	layers = new Layer*[layer_count];
	layer_length = new std::size_t[layer_count];

	auto indication = list.end() - 1;
	std::size_t tmp = 1;
	// 从最后一层逐层建立
	while (indication > list.begin()) {
		layers[layer_count - tmp] =
			new Layer((*(indication - 1))->size, (*indication)->size,
					  *(*indication)->activation);
		layer_length[layer_count - tmp] = (*indication)->size;
		--indication;
		++tmp;
	}

	layers[0] = new Layer(0, (*indication)->size, *((*indication)->activation));

	// 释放LayerConstructionInfo
	for (auto i : list) delete i;
}

Network::~Network() {
	for (std::size_t i = 0; i < layer_count; ++i) delete layers[i];
	delete[] layers;
	delete[] layer_length;
}

LayerConstructionInfo* Network::layer(std::size_t size, IFunction& func) {
	// 注意，new后需要delete！
	LayerConstructionInfo* re = new LayerConstructionInfo;
	re->size = size;
	re->activation = &func;

	return re;
}

void Network::print_network(ostream& out) const {
	for (std::size_t i = 0; i < layer_count; ++i)
		out << "Layer#" << i << std::endl
			<< *(layers[i]) << std::endl
			<< "-------" << std::endl;
}

const double* Network::evaluate(node_type data[]) {
	layers[1]->set_input(data);
	for (std::size_t i = 1; i < layer_count - 1; ++i) {
		layers[i]->forward_propagation(true);
		layers[i + 1]->set_input(layers[i]->get_val());
	}
	layers[layer_count - 1]->forward_propagation(true);
	return layers[layer_count - 1]->get_val();
}

const node_type* Network::get_result() {
	return layers[layer_count - 1]->get_val();
}

void Network::single_fit(double learning_rate, node_type target[]) {
	node_type* last_delta = new node_type[layer_length[layer_count - 1]];

	for (std::size_t i = 0; i < layer_length[layer_count - 1]; ++i)
		last_delta[i] = (layers[layer_count - 1]->get_val()[i] - target[i]) *
						layers[layer_count - 1]->get_IFunction().d_activation(
							layers[layer_count - 1]->get_integeration()[i]);

	layers[layer_count - 1]->set_ddelta(last_delta);

	// CLF 修改
	layers[layer_count - 1]->back_propagation(nullptr, learning_rate);

	for (std::size_t i = layer_count - 2; i >= 1; --i) {
		layers[i]->back_propagation(layers[i + 1], learning_rate);
	}

	delete[] last_delta;
}
