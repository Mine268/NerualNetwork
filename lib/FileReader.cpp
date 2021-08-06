//
// Created by xzhy324 on 2021/8/3.
//

#include "FileReader.h"

FileReader::FileReader(int classification, const string& dataPath, const string& labelPath)
	: data_ifs(dataPath.c_str(), ios::binary),
	  label_ifs(labelPath, ios::binary) {
	// 读取image集的magic number作校验
	char buf[1000];
	data_ifs.read(buf, 4);
	int magic = MsbInt(buf);
	if (magic != 0x00000803) {
		cerr << "incorrect data file magic number" << endl;
	}
	//获取image集样本信息
	data_ifs.read(buf, 4);
	data_n = MsbInt(buf);
	data_ifs.read(buf, 4);
	data_row = MsbInt(buf);
	data_ifs.read(buf, 4);
	data_col = MsbInt(buf);
	data_size = data_row * data_col;

	// 读取label集的magic number作校验
	label_ifs.read(buf, 4);
	magic = MsbInt(buf);
	if (magic != 0x00000801) {
		cerr << "incorrect label file magic number" << endl;
	}
	//获取label集样本信息
	label_ifs.read(buf, 4);
	label_n = MsbInt(buf);
	label_size = classification;

	if (label_n != data_n) {
		cerr << "labels and images are unmatched!" << endl;
	}

	//初始化文件信息
	FileInfo.image_n = data_n;
	FileInfo.label_size = label_size;
	FileInfo.image_size = data_size;
}

int FileReader::MsbInt(const char* buf, int len) {	//将高位表示转化为低位表示
	int base = 1;
	int ret = 0;
	for (int i = len - 1; i >= 0; i--) {
		ret += (unsigned char)buf[i] * base;
		base *= 256;
	}
	return ret;
}

data_type* FileReader::getLabel() {
	if (!label) delete label;
	label = new data_type[label_size];
	for (int i = 0; i < label_size; ++i) label[i] = .0;
	label[label_ifs.get()] = 1.;
	return label;
}

data_type* FileReader::getData() {
	if (!data) delete data;
	data = new data_type[data_size];
	for (int i = 0; i < data_size; i++) {
		data[i] = data_ifs.get();
	}
	return data;
}
