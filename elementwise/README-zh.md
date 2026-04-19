# The introduction of elementwise kernel
elementwise指的是需要逐位进行运行的操作，例如向量A, B的逐元素相加、相乘，常见的操作有add, Sigmoid, ReLU等，是典型的Memory Bound Kernel。
首先分析一下 elementwise 的计算访存比, 假设做逐元素处理的两个矩阵的形状都是N * N, 总计算量是 N^2; 我们需要读两个这样的矩阵, 写一个这样的矩阵, 以 4Bytes 的标准 float 存储, 总访存是 4 * (3N^2), 可以算出, 计算访存比是 1 / 12, 计算强度极低。
单纯的elementwise操作并没有太多优化技巧，只要正常地设置block和thread，并使用好FLAOT2 / FLOAT4向量化访存，就可以达到90%+的性能。
进行操作的算子融合，一次读写处理尽可能多的数据是常见的优化方案，在这里只写一个Add算子难免有些无趣，因此，我在此尝试一个初始SwiGLU的算子优化，其公式如下: 
$\mathrm{SwiGLU}(x) = \mathrm{Swish}(xW_1 + bias_1) \odot (xW_2 + bias_2)$
$\mathrm{Swish}(x) = x \cdot {Sigmoid}(x) = \frac{x}{1+e^{-x}}$

我们看公式的计算逻辑顺序，首先要算出内层的矩阵乘，加偏置后，第一部分先Swish，再和第二部分逐元素乘，其中的Swish和Hadamard积是Elementwise, xW是Gemm, 二者有前后依赖关系, 且计算Switsh对计算机来说很容易，不是很好优化. 
这里的Gemm计算偷了个懒，没有做数据预取，需要学习这部分内容的移步sgemm folder

## How to compile 
```bash
nvcc -std=c++17 -O2 elementwise.cu -o elementwise_test
# optional: -O3, -arch=sm_XX for your GPU
./elementwise_test
```