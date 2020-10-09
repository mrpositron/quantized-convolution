all: dnn_cuda.o convolution.o convolution

dnn_cuda.o: dnn_cuda.cu
	nvcc  -O3 -c dnn_cuda.cu

convolution.o: convolution.c
	gcc  -O3 -c convolution.c 

convolution:  dnn_cuda.o convolution.o
	nvcc -O3 -o convolution convolution.o dnn_cuda.o

clean:
	rm -f convolution convolution.o dnn_cuda.o