# Quantized Covolution


The purpose of this project is to compare different means of computing convolution operation, and see if naive quantiization actually speed ups operation.


## Table of Contents:
0. Data
1. Naive implementation of convolution in C language.
2. Quantizing the naive implemetation of convolution using lower precisions.
3. Applying CPU vectorization using AVX instructions and Pthreads.
4. GPU vectorization using CUDA.
5. Analysis.
6. References.

## 0. Data

All experimets were performed on three combinations of input and kernel tensors. Download the following data.zip from this [link](https://drive.google.com/file/d/1kVrZ03pIykRpTehYglW8ykRLqIdr2JEk/view?usp=sharing).

The structure of the data folder:

* data.zip
  * sample 1
    * input_tensor.bin
    * kernel_tensor.bin
  * sample 2
    * input_tensor.bin
    * kernel_tensor.bin
  * sample 3
    * input_tensor.bin
    * kernel_tensor.bin

## 1. Naive implementation of convolution operation

The algorithm to perform convolution was impemented using im2col technique mentioned in [1]. Simply speaking, transformations to the input and kernel matrices are made so that only one dot product is needed to compute convolution. The code implementation and analysis can be found in the folder prob1.

## 2. Quantizing the naive implementation of Convolution using lower precision data types.

Using lower precision values can greatly improve the running time of the algorithm. The reason for that is that there will be less data movement from the memory. The code with the corresponding makefile and analysis can be found in the folder prob2.

## 3. Applying CPU vectorization using AVX instructions and Pthreads

Using AVX and Pthreads siginificant speed up was achieved. The source code with corresponding makefile can be found in the folder prob3.

## 4. GPU vectorization using CUDA

Conovolution implemented using CUDA is also presented. The corresponding .cu file and makefile can be found in prob4.

## 5. Analysis

<p align="center">
  <img width="283" height="577" src="https://github.com/MrPositron/Quantized-Covolution/blob/main/analysis1.png">
</p>

1. From the figure above it can be clearly seen that pthread with AVX instructions using 16 bit integers performs better than other data types. However the margin is really small. Thus, if you will use pthreads with AVX instructions, it is better to use floating numbers. Because it will have no loss in accuracy.
If there are no opportunities to use pthreads with avx instructions then it is preferrable to use 8 bit integers. However, it should be noted that in an 8 bit integer implementation, 16 bit integers were used to accumulate the sum. Therefore, it is not pure 8 bit integer implementation. This implementation leads to 2x speedup over 32 bit floating numbers with a slight loss in accuracy. There are two CUDA lines. One represents total time, and second represents time only for CUDA multiplication (without transferring data from host to CUDA). Thus, it can be clearly seen that exchanging data is a bottleneck for the implementation with a CUDA. It is interesting to notice how fast matrix multiplication is without considering data movements between storages. Another interesting point is that time spent with CUDA is nearly the same with different sizes of input and kernel.

2. In order to choose the optimal implementation, factors that determine efficiency have to be understood. There are several factors but most important ones are memory, speed and energy consumption. Using lower precision numbers will consume less memory. Furthermore, it will make memory access faster. Multiplication of lower precisions values is also faster. Less memory accesses on the other hand will require less energy. Thus, it can be clearly seen why quantization is efficient.

3. Running several threads may be inefficient in terms of energy. Thus, it may be better to use numbers with lower precision, especially if we want to use it on low-power mobile devices.

4. From my personal perspective, the naive quantization is justified. However, there should be a clever way to choose constants. Choosing optimal constants is a really arduous process. The process for searching them is non-linear. For example, increasing the scaling constant does not always decrease NRMS error. There are several works that suggest using Reinforcement Learning agents to search for the optimal quantization bit numbers [2]. Naive quantization can be sub-optimal. Nevertheless, it shows that inference time can be decreased without hurting the accuracy of the network much. In my opinion, acceptable NRMSE is around ~0.01.

## 6. References

[1] - https://cs231n.github.io/convolutional-networks/

[2] - AutoQ: Automated Kernel-Wise Neural Network Quantization. Qian Lou, Feng Guo, Lantao Liu, Minje Kim, Lei Jiang.
