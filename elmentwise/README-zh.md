# The introduction of element wise kernel
elementwise指的是需要逐位进行运行的操作，例如向量A, B的逐元素相加、相乘，常见的操作有add, Sigmoid, ReLU等，是典型的Memory Bound Kernel
单纯的elementwise操作并没有太多优化技巧，只要正常地设置block和thread，并使用好FLAOT2 / FLOAT4向量化访存，就可以达到90%+的性能

## Fused kernel
进行操作的算子融合，一次读写处理尽可能多的数据是常见的优化方案，在这里只写一个Add算子难免有些无趣，因此，我在此尝试一个初始SwiGLU的算子优化，公式如下: 
$\mathrm{SwiGLU}(x) = \mathrm{Swish}(xW_1 + bias_1) \odot (xW_2 + bias_2)$
$\mathrm{Swish}(x) = x \cdot {Sigmoid}(x) = \frac{x}{1+e^{-x}}$

我们看公式的计算逻辑顺序，首先要算出内层的矩阵乘，加偏置后，第一部分先Swish，再和第二部分逐元素乘，其中的Swish和Hadamard积是Elementwise, xW是Gemm, 二者有前后依赖关系, 且计算Switsh对计算机来说很容易，不是很好优化. 
这里的Gemm计算偷了个懒，没有做数据预取，需要学习这部分内容的移步sgemm folder

## How to compile 
nvcc -std=c++17 elementwise/elementwise.cu -o elementwise/elementwise_test
-O2 -O3优化可以自选, -arch指定自己的设备型号也请自选