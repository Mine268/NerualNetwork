//
// Created by God on 2021/7/13.
//

#include "Layer.h"
#include <random>
#include <iterator>


void Layer::new_data() {
    //申请相应的空间
    weight = new node_type *[input_size];
    dweight = new node_type *[input_size];
    for (int i = 0; i < input_size; i++) {
        weight[i] = new node_type[output_size];
        dweight[i] = new node_type[output_size];
    }
    biases = new node_type[output_size];
    value = new node_type[output_size];
    integeration = new node_type[output_size];
    ddelta = new node_type[output_size];
    preval = new node_type[input_size];
    dbiases = new node_type[output_size];
}

Layer::Layer(int _inputsize, int _outputsize, IFunction& function):activation(function) {
    input_size = _inputsize;
    output_size = _outputsize;
    new_data();
    init_wb();
}

Layer::Layer(const Layer & one):activation(one.activation) {
    //复制构造函数不会复制数据，而只是初始化其大小，选择的激活函数与传入的对象相同
    input_size = one.input_size;
    output_size = one.output_size;
    new_data();
    init_wb();
}

Layer::~Layer() {
    for (int i = 0; i < input_size; i++) {
        delete[]weight[i];
        delete[]dweight[i];
    }
    delete[]biases;
    delete[]value;
    delete[]integeration;
    delete[]ddelta;
    delete[] preval;
    delete[] dbiases;
}

IFunction & Layer::get_IFunction() {
    //返回当前层的激活函数
    return activation;
}

void Layer::init_wb() {
    //初始化weight矩阵与biases,biases全部初始化为0，weight初始化为均值为0，方差为1的矩阵
    //初始化weight的过程中顺带把dweight初始化为0
    //    auto seed = 3;//固定值用来debug
    auto seed = rand()%1000;
    default_random_engine engine_need(seed);
    normal_distribution<node_type> distribution(0,1);
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < output_size; j++) {
            weight[i][j] = distribution(engine_need);
            dweight[i][j] = 0;
        }
    }
    for (int i = 0; i < output_size; i++) {
        biases[i] = 0;
    }
}


void Layer::set_input(const node_type *input) {
    //将传入的数据保存在inputdata中，方便前向传播使用
    //对于第一层来说，就是传入训练集数据，对于后面的层来说，要使用read_output获取前一层的value然后传入input_data中
    copy(input, input + input_size, preval);
}

const node_type *Layer::get_val(){
    //返回当前层的激活值
    return value;
}

const node_type *Layer::get_integeration() {
    return integeration;
}

void Layer::read_output() const {
    cout << "values: \n";
    for_each(value, value + output_size, [](node_type val) { cout << val << " "; });
    cout << endl;
}


ostream &operator<<(ostream &os, Layer &one) {
    //打印层的相关信息，包括其大小，weight，biases
    os << "input_size: " << one.input_size << endl;
    os << "output_size: " << one.output_size << endl;
    os << "weight: " << endl;
    for (int i = 0; i < one.input_size; i++) {
        for (int j = 0; j < one.output_size; j++) {
            cout << one.weight[i][j] << " ";
        }
        cout << endl;
    }
    os << "biases: " << endl;
    for_each(one.biases, one.biases + one.output_size, [](node_type val) { cout << val << " "; });
    cout << endl;
    return os;
}

void Layer::forward_propagation(bool useactfunction) {
    for (int i = 0; i < output_size; i++) {
        node_type z = biases[i];
        for (int j = 0; j < input_size; j++) {
            z += preval[j] * weight[j][i];
        }
        integeration[i] = z;
        if (useactfunction)
            value[i] = activation.activation(z);
        else value[i] = z;
    }
}

void Layer::set_ddelta(node_type *delta) {
    //保存传入的detal,用于最后一层，因最后一层的损失函数需要在Net层计算，需要传入最后一层
    copy(delta, delta + output_size, ddelta);
}


void Layer::back_propagation(Layer &preLayer,node_type learning_rate) {
    //传入后一层，并根据后一层的ddelta，求当前层的ddelta,再求出dweight与dbiases
    //第一步，求出当前层的ddelta
    for (int i = 0; i < output_size; i++) {
        node_type sum = 0;
        for (int k = 0; k < preLayer.output_size; k++) {
            sum += preLayer.ddelta[k] * preLayer.weight[i][k] * activation.d_activation(integeration[i]);
        }
        ddelta[i] = sum;
    }
    //第二步，求ddweight
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < output_size; ++j) {
            dweight[i][j] = ddelta[j] * preval[i];
            weight[i][j] -= (learning_rate*dweight[i][j]);
        }
    }
    //第三步，求dbiases;
    for (int i = 0; i < output_size; i++) {
        dbiases[i] = ddelta[i];
        biases[i] -= (learning_rate*dbiases[i]);
    }
}


void Layer::debug() {
    cout << "integration: \n";
    for_each(integeration, integeration + output_size, [](node_type val) { cout << val << " "; });
    cout << endl;
    cout << "inputdata: \n";
    for_each(preval, preval + input_size, [](node_type val) { cout << val << " "; });
    cout << endl;
}

void Layer::debug2() {
    cout << "ddelta: \n";
    for_each(ddelta, ddelta + output_size, [](node_type val) { cout << val << " "; });
    cout << endl;
}