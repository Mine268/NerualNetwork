#include "FileReader.h"

int main() {
	FileReader fr("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	for (int n = 1; n <= fr.FileInfo.image_n; n++) {
		cout << "label:" << *(fr.getLabel()) << endl;
		cout << "data:";
		data_type *data = fr.getData();
		for (int j = 0; j < fr.FileInfo.image_size; j++) cout << data[j] << " ";
		cout << endl;
	}
}