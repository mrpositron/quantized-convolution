#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "dnn_cuda.h"
#include <sys/time.h>

#define ij2index(i,j,W) (i*W + j)
#define ijk2index(i,j,k,W,C) ((W * i + j)*C + k)
#define ijko2index(i,j,o,k,W,IC,OC) (((W * i + j )*OC + o)*IC + k)

int main(int argc, char * argv[]){
	FILE *ptr;
	int read;
	ptr = fopen(argv[1], "rb"); 
	if (!ptr){
		printf("unable to open file\n");
		return 1;
	}
	int input_dims[4], N, H, W, C;
	read = fread(input_dims, sizeof(input_dims), 1, ptr);
	N = input_dims[0]; /*batch size, usually 1*/
	H = input_dims[1]; /*input tensor height*/
	W = input_dims[2]; /*input tensot width*/
	C = input_dims[3]; /*input tensor channels*/
	float* input_data;
	input_data = (float *) malloc(N * H * W * C * sizeof(float));
	read = fread(input_data, sizeof(float), N*H*W*C, ptr);
	fclose(ptr);
	ptr = fopen(argv[2], "rb"); 
	if (!ptr){
		printf("unable to open file\n");
		return 1;
	}
	int kernel_dims[4];
	read = fread(kernel_dims, sizeof(kernel_dims), 1, ptr);
	int KH, KW, OC, IC;
	KH = kernel_dims[0]; /**/
	KW = kernel_dims[1];
	OC = kernel_dims[2];
	IC = kernel_dims[3];
	float* kernel_data;
	kernel_data = (float *) malloc(KH * KW * OC * IC * sizeof(float));
	read = fread(kernel_data, sizeof(float), KH * KW * OC * IC, ptr);
	fclose(ptr);
	struct timeval stop, start;

	int ii,jj, PD, HP, WP;
	PD = KH/2; /*pading size*/
	HP = (H + 2 * PD); /*padded height*/
	WP = (W + 2 * PD); /*padded width*/
	float* padded; /*padded tensor*/
	padded = (float *) malloc(N * HP * WP * C * sizeof(float*));
	/* filling padded tensor with zeros*/
	for (int h = 0; h < N * HP * WP * C; h++){
		padded[h] = 0.0;
	}
	/*------------------------------------------------------*/
	
	ii = 0;
	for (int i = PD; i < HP - PD; i++){
		jj = 0;
		for (int j = PD; j < WP - PD; j++){
			for (int k = 0; k < C; k++){
				padded[ijk2index(i,j,k,WP,C)] = input_data[ijk2index(ii,jj,k,W,C)];
			}
			jj++;
		}
		ii++;
	}
	int HO,WO,ix, jx, kh, kw, r;
	kh = KH/2; /*half of a kernel size*/
	kw = KW/2; /*half of a kernel size*/
	HO = HP - KH + 1; /* output height */
	WO = WP - KW + 1; /* output width  */
	/* im2col shouldbe of (WO*HO,IC* KH * KW) */
	/* below it is in a row-major format */
	float* im2col;
	im2col = (float *) malloc(WO * HO * KH * KW * IC * sizeof(float*));
	ix = 0;
	for( int i = kh; i < HP - kh; i++){
		jx = 0;
		for (int j = kw; j < WP - kw; j++){
			r = 0;
			for (int xx = i - kh; xx < i + kh + 1; xx++){
				for (int yy = j - kh; yy < j + kh + 1; yy++){
					for (int k = 0; k < IC; k++){
						im2col[(KH*KW*IC)*(ix*WO + jx)  + r] = padded[ijk2index(xx,yy,k,WP,IC)];
						r++;
					}
				}
			}
			jx++;
		}
		ix++;
	}
	float* kernel2col;
	kernel2col = (float *) malloc(KW* KH* IC * OC * sizeof(float*));
	int row;
	row = 0;
	for (int i = 0; i < KH; i++){
		for (int j = 0; j < KW; j++){
			for (int k = 0; k < IC; k++){
				for (int o = 0; o < OC; o++){
					kernel2col[ij2index(row, o, OC)] = kernel_data[ijko2index(i,j,o,k,KW,IC,OC)];
				}
				row++;
			}
		}
	}

	int m;
	int n;
	int k;
	m = WO*HO;
	k = IC* KH * KW;
	n = OC;
	float* c;
	c = (float *) malloc(m*n * sizeof(float));
	gettimeofday(&start, NULL);
	matrix_mul(im2col, kernel2col,c, m, n, k);

	gettimeofday(&stop, NULL);
	printf("Inference time is %lu microseconds\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
	
	ptr = fopen("output_tensor.bin", "w+");
	int output_size[4] = {N, HO, WO, OC};
	fwrite(output_size, sizeof(output_size), 1, ptr);
	fwrite(c, sizeof(float),N*HO*WO*OC, ptr);
	fclose(ptr);

	free(c);
	free(input_data);
	free(kernel_data);
	free(im2col);
	free(kernel2col);
	return 0;
}