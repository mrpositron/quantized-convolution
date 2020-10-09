all:  mulFP32.o mulINT32.o mulINT16.o convolution.o convolution

mulFP32.o: mulFP32.c
	gcc  -O3 -c mulFP32.c -lpthread -mavx2 -lm

mulINT32.o: mulINT32.c
	gcc  -O3 -c mulINT32.c -lpthread -mavx2 -lm

mulINT16.o: mulINT16.c
	gcc  -O3 -c mulINT16.c -lpthread -mavx2 -lm

convolution.o: convolution.c
	gcc  -O3 -c convolution.c -lpthread -mavx2 -lm

convolution: convolution.o mulFP32.o mulINT32.o
	gcc -O3 -o convolution  convolution.o mulFP32.o mulINT32.o mulINT16.o -lpthread -mavx2 -lm

clean:
	rm -f convolution convolution.o mulFP32.o mulINT32.o mulINT16.o