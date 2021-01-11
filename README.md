# Quantized Covolution


In this project we perform convolution operation.


Table of Contents:
1. Naive implementation of convolution in C
2. Quantizing the naive implemetation of convolution using lower precisions.
3. Applying CPU vectorization using AVX instructions and Pthreads
4. GPU vectorization using CUDA
5. Analysis


## Naive implementation of convolution operation

The algorithm that I used to perform convolution was im2col mentioned in [1]. Simply speaking we transform the convolution operation using one big dot product.

The code implementation can be found in the folder prob1.

## Quantizing the naive implementation of Convolution using lower precision data types.

Using lower precision values can greatly improve the running time of the algorithm. The reason for that is that there will be less data movement from the memory. The code with the corresponding makefile can be found in the folder prob2.

## Applying CPU vectorization using AVX instructions and Pthreads

By applying vectorization using AVX and Pthreads we can speed up the algorithm. The source code with corresponding makefile can be found in the folder prob3.

## GPU vectorization using CUDA

Finally, I implemented the convolution using CUDA. The corresponding .cu file and makefile can be found in prob4.

## Analysis


