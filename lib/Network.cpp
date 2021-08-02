#include "Network.h"

Network::Network(std::initializer_list<LayerConstructionInfo*> list)
	: layers(nullptr), layer_size(0) {
	// 首先计算出总共需要多少层
	for (auto lyrInfo : list) ++layer_size;

	// 分配空间
	layers = new Layer*[layer_size];

	auto indication = list.end();
	std::size_t tmp = 1;
    // 最后一层建立
	layers[layer_size - tmp] =
		new Layer((*indication)->size, 0, *(*indication)->activation);
	--indication;
	--tmp;
    // 从最后一层逐层建立
	while (indication >= list.begin())
		layers[layer_size - tmp] =
			new Layer((*indication)->size, (*(indication + 1))->size,
					  *(*indication)->activation);
    
    // 释放LayerConstructionInfo
    for (auto i : list)
        delete i;
}

Network::~Network() {
    Layer * head = layers[0];
    for (std::size_t i = 0; i < layer_size; ++i)
        delete (head + i);
}

LayerConstructionInfo* Network::layer(std::size_t size, IFunction& func) {
	// 注意，new后需要delete！
	LayerConstructionInfo* re = new LayerConstructionInfo;
	re->size = size;
	re->activation = &func;

	return re;
}
