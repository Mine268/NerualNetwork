//
// Created by xzhy324 on 2021/8/3.
//

#ifndef NERUALNETWORK_FILEREADER_H
#define NERUALNETWORK_FILEREADER_H
typedef int data_type;	//训练集基本元素类型

#include <fstream>
#include <iostream>
#include <string>

using namespace std;

class FileReader {
   public:
	FileReader(int classification, const string& dataPath, const string& labelPath);
	~FileReader();
	data_type* getLabel();
	data_type* getData();
	struct {
		int image_n;	 // 样本数
		int image_size;	 // 单个image向量的长度
		int label_size;	 // 单个label向量的长度
	} FileInfo;	 // 通过公有的FileInfo来和类外进行一些文件基本信息的传递
   private:
	int data_n;			 // 图像数量
	int data_col;		 // image列数
	int data_row;		 // image行数
	int data_size;		 // 返回data类型向量的长度，为row * col
	int label_n;		 // 标签数量，设置两个量以便于校验
	int label_size;		 // 返回label类型向量的长度
	ifstream data_ifs;	 // 图像文件流
	ifstream label_ifs;	 // 标签文件流
	data_type* label;
	data_type* data;
	static inline int MsbInt(
		const char buf[],
		int len = 4);  //用于将mnist原始数据中的高位表示转化为低位表示
};

#endif	// NERUALNETWORK_FILEREADER_H
