#include "FileReader.h"

int main(){
    FileReader fr("t10k-images.idx3-ubyte","t10k-labels.idx1-ubyte");
    for(int i=1;i<=fr.FileInfo.image_n;i++){
        cout<<"label:"<<*(fr.getLabel())->begin()<<endl;
        cout<<"data";
        for(auto hui_du_zhi : *fr.getData()){
            cout<< hui_du_zhi;
        }
        cout<<endl;
    }
}