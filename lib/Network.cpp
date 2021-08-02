#include "Network.h"

Network::Network(std::initializer_list<LayerConstructionInfo*> list)
	: layers(nullptr), layer_count(0) {
	// 首先计算出总共需要多少层
	for (auto lyrInfo : list) ++layer_count;

	// 分配空间
	layers = new Layer*[layer_count];
	layer_length = new std::size_t[layer_count];

	auto indication = list.end();
	std::size_t tmp = 1;
	// 最后一层建立
	layers[layer_count - tmp] =
		new Layer((*indication)->size, 0, *(*indication)->activation);
	layer_length[layer_count - tmp] = (*indication)->size;
	--indication;
	++tmp;
	// 从最后一层逐层建立
	while (indication >= list.begin()) {
		layers[layer_count - tmp] =
			new Layer((*indication)->size, (*(indication + 1))->size,
					  *(*indication)->activation);
		layer_length[layer_count - tmp] = (*indication)->size;
		--indication;
		++tmp;
	}

	// 释放LayerConstructionInfo
	for (auto i : list) delete i;
}

Network::~Network() {
	Layer* head = layers[0];
	for (std::size_t i = 0; i < layer_count; ++i) delete (head + i);
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
