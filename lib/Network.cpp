#include "Network.h"

Network::Network(std::initializer_list<LayerConstructionInfo*> list)
	: layers(nullptr), layer_count(0), layer_length(nullptr) {
	// ���ȼ�����ܹ���Ҫ���ٲ�
	for (auto lyrInfo : list) ++layer_count;

	// ����ռ�
	layers = new Layer*[layer_count];
	layer_length = new std::size_t[layer_count];

	auto indication = list.end() - 1;
	std::size_t tmp = 1;
	// �����һ����㽨��
	while (indication > list.begin()) {
		layers[layer_count - tmp] =
			new Layer((*(indication - 1))->size, (*indication)->size,
					  *(*indication)->activation);
		layer_length[layer_count - tmp] = (*indication)->size;
		--indication;
		++tmp;
	}

	layers[0] = new Layer(0, (*indication)->size, *((*indication)->activation));
	layer_length[0] = (*indication)->size;

	// �ͷ�LayerConstructionInfo
	for (auto i : list) delete i;
}

Network::~Network() {
	for (std::size_t i = 0; i < layer_count; ++i) delete layers[i];
	delete[] layers;
	delete[] layer_length;
}

LayerConstructionInfo* Network::layer(std::size_t size, IFunction& func) {
	// ע�⣬new����Ҫdelete��
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

	// CLF �޸�
	layers[layer_count - 1]->back_propagation(nullptr, learning_rate);

	for (std::size_t i = layer_count - 2; i >= 1; --i) {
		layers[i]->back_propagation(layers[i + 1], learning_rate);
	}

	delete[] last_delta;
}

void Network::fit(double learning_rate, int epoch,
				  int batch_size, std::string data_path,
				  std::string label_path) {
	FileReader fr(layer_length[layer_count - 1], data_path, label_path);
	auto basic_info = fr.FileInfo;
	bool legal1, legal2;

	// ������Ϣ����
	legal1 = (basic_info.image_size == layer_length[0]);
	legal2 = (basic_info.label_size == layer_length[layer_count - 1]);

	if (!(legal1 & legal2)) {
		std::cerr << "�����ļ����ǩ�ļ���ģ�Ͳ�һ�¡�" << std::endl;
		if (!legal1)
			std::cerr << "�����ļ���ģ�Ͳ�һ�¡��ļ���ַ��" << data_path
					  << std::endl;
		if (!legal2)
			std::cerr << "��ǩ�ļ���ģ�Ͳ�һ�¡��ļ���ַ��" << label_path
					  << std::endl;
		return;
	} else {
		std::cout << "��ʼѵ����" << std::endl
				  << "�����ļ���" << data_path << std::endl
				  << "��ǩ�ļ���" << label_path << std::endl;

		// delta�洢����
		node_type *last_delta = new node_type[basic_info.label_size];
		node_type *d_data_ptr = new node_type[basic_info.image_size],
				  *d_label_ptr = new node_type[basic_info.label_size];
		data_type *i_data_ptr, *i_label_ptr;

		for (std::size_t turn_num = 0; turn_num < epoch; ++turn_num) {
			// ��ʼѵ����turn_num��
			std::cout << "��ʼѵ��Epoch " << turn_num << "...";

			for (int group_num = 0;
				 (group_num + 1) * batch_size < basic_info.image_n;
				 ++group_num) {
				// last_delta��0
				for (int i = 0; i < basic_info.label_size; ++i)
					last_delta[i] = 0.;
				// һ��ȡbatch_size��ƽ���ٷ��򴫲�
				for (int sample_num = 0; sample_num < batch_size;
					 ++sample_num) {
					// ����ԭʼ���ݣ�����Ϊdata_type
					i_data_ptr = fr.getData();
					i_label_ptr = fr.getLabel();

					// ��������ת��
					// TODO: ����ʹ�ô��뺯��ָ�����ת��
					for (int i = 0; i < basic_info.image_size; ++i)
						d_data_ptr[i] = (node_type)i_data_ptr[i] / 256.;
					for (int i = 0; i < basic_info.label_size; ++i)
						d_label_ptr[i] = (node_type)i_label_ptr[i];

					// ����ǰ�򴫲�
					auto result = this->evaluate(d_data_ptr);
					// �������һ��delta
					for (int i = 0; i < basic_info.label_size; ++i)
						last_delta[i] +=
							(result[i] - d_label_ptr[i]) *
							layers[layer_count - 1]
								->get_IFunction()
								.d_activation(layers[layer_count - 1]
												  ->get_integeration()[i]);
				}
				// deltaȡƽ��
				for (int i = 0; i < basic_info.label_size; ++i)
					last_delta[i] /= (node_type)batch_size;

				// ���з��򴫲�
				layers[layer_count - 1]->set_ddelta(last_delta);
				layers[layer_count - 1]->back_propagation(nullptr,
														  learning_rate);
				for (std::size_t i = layer_count - 2; i >= 1; --i)
					layers[i]->back_propagation(layers[i + 1], learning_rate);
			}
			// �����Ϣ
			std::cout << "ѵ�����" << std::endl;
		}
		delete[] last_delta;
		delete[] d_data_ptr;
		delete[] d_label_ptr;
	}
}
