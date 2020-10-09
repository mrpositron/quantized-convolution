#include <stdio.h>
#include <sys/time.h>

extern "C" {
#include "dnn_cuda.h"
}

__global__ void multiply(float *a, float *b, float*c, int m, int n, int k)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0;

    if ( col < n && row < m){
        for (int i = 0; i < k; i++){
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
        
    }
}

void matrix_mul(float *a, float *b, float *c, int m, int n, int k)
{
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc( (void**)&dev_a, (m*k) * sizeof(float) ) ;
    cudaMalloc( (void**)&dev_b, (k*n) * sizeof(float) );
    cudaMalloc( (void**)&dev_c, (m*n) * sizeof(float) );
    
    cudaMemcpy( dev_a, a, (m*k) * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, (k*n) * sizeof(float), cudaMemcpyHostToDevice );

    int bx = 512;
    dim3 blockDim(bx, bx);
    int gx, gy;
    if (m % bx == 0) gx = (m/bx);
    else gx = (m/bx) + 1;

    if (n % bx == 0) gy = (n/bx);
    else gy = (n/bx) + 1;
    dim3 gridDim(gx, gy);
    struct timeval stop, start;
    gettimeofday(&start, NULL);
    multiply<<<blockDim,gridDim>>>(dev_a, dev_b, dev_c, m, n, k);
    cudaDeviceSynchronize();
    gettimeofday(&stop, NULL);
    printf("CUDA Inference time is %lu microseconds\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
    cudaMemcpy( c, dev_c, (m*n) * sizeof(float), cudaMemcpyDeviceToHost )   ;
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );
}