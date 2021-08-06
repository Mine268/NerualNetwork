//
// Created by xzhy324 on 2021/8/3.
//

#ifndef NERUALNETWORK_FILEREADER_H
#define NERUALNETWORK_FILEREADER_H
<<<<<<< HEAD

=======
typedef int data_type;//训练集基本元素类型

#include <iostream>
>>>>>>> dev_file
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

class FileReader {
<<<<<<< HEAD
   public:
	FileReader(const string& dataPath, const string& labelPath);
	vector<int>* getLabel();
	vector<int>* getData();
	struct {
		int image_n;
		int image_row;
		int image_col;
	} FileInfo;	 //通过公有的FileInfo来和类外进行一些文件基本信息的传递
   private:
	int data_n;			 //图像数量
	int label_n;		 //标签数量，设置两个量以便于校验
	int data_col;		 // image列数
	int data_row;		 // image行数
	ifstream data_ifs;	 //图像文件流
	ifstream label_ifs;	 //标签文件流
	vector<int> label;
	vector<int> data;
	static inline int MsbInt(
		const char buf[],
		int len = 4);  //用于将mnist原始数据中的高位表示转化为低位表示
=======
public:
    FileReader(const string& dataPath,const string& labelPath);
    data_type* getLabel();
    data_type* getData();
    struct{
        int image_n;//样本数
        int image_size;//单个image向量的长度
        int label_size;//单个label向量的长度
    }FileInfo;//通过公有的FileInfo来和类外进行一些文件基本信息的传递
private:
    int data_n;//图像数量
    int data_col;//image列数
    int data_row;//image行数
    int data_size;//返回data类型向量的长度，为row * col
    int label_n;//标签数量，设置两个量以便于校验
    int label_size;//返回label类型向量的长度
    ifstream data_ifs;//图像文件流
    ifstream label_ifs;//标签文件流
    data_type *label;
    data_type *data;
    static inline int MsbInt(const char buf[],int len=4);//用于将mnist原始数据中的高位表示转化为低位表示
>>>>>>> dev_file
};

#endif	// NERUALNETWORK_FILEREADER_H
