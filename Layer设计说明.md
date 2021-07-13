# class Layer

## 成员变量

- `size:int` 表示节点的个数
- `value:NTYPE[]` 表示每个节点的激活值
- `integeration:NTYPE[]` 表示每个节点的整合函数
- `biases:NTYPE[]` 偏置
- `weights:NTYPE[][]` 权重（到这一层）
- `ddelta:NTYPE[]` 中间参数
- `activation:IFunction` 激活函数接口（使用预设）
- `nextLayer:Layer` 下一层
- `prevLayer:Layer` 上一层

## 成员函数

### 构造函数

- `Layer(int size, cosnt Layer * pre)`

  构造函数，用来创建层，size表示这个层的节点数量，Layer表示前一层。

- `Layer(int size, const Layer * pre, IFunction activation=sigmoid)`

  构造函数，用于创建层，size表示这个层的节点数量，activation表示这一层的激活函数，activation默认为sigmoid。

- `~Layer()`

  析构函数，释放数组资源。

- `void set_input(NTYPE array[])`

  将array中的元素依次复制为该层的输入。

- `const NTYPE[] read_output`

  将当前层的结果输出。

- `void forward_propagation()`

  前向传播，依据前一层的激活值计算本层的激活值。

- `void back_propagation()`

  反向传播，依据后一层的delta计算这一层的delta并更新这一层的参数。