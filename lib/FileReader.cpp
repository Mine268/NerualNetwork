//
// Created by xzhy324 on 2021/8/3.
//

#include "FileReader.h"

FileReader::FileReader(const string& dataPath, const string& labelPath)
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

	// 读取label集的magic number作校验
	label_ifs.read(buf, 4);
	magic = MsbInt(buf);
	if (magic != 0x00000803) {
		cerr << "incorrect label file magic number" << endl;
	}
	//获取label集样本信息
	label_ifs.read(buf, 4);
	label_n = MsbInt(buf);

	if (label_n != data_n) {
		cerr << "labels and images are unmatched!" << endl;
	}

	//初始化文件信息
	FileInfo.image_n = data_n;
	FileInfo.image_row = data_row;
	FileInfo.image_col = data_col;
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

vector<int>* FileReader::getLabel() {
	label.clear();
	label.push_back(label_ifs.get());
	return &label;
}

vector<int>* FileReader::getData() {
	data.clear();
	for (int i = 0; i < data_row * data_col; i++) {
		data.push_back(data_ifs.get());
	}
	return &data;
}
