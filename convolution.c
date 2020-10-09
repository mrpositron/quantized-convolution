/* matrix multiplication with im2col16 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <inttypes.h>
#include <math.h>
#include "mulFP32.h"
#include "mulINT32.h"
#include "mulINT16.h"

#define ij2index(i,j,W) (i*W + j)
#define ijk2index(i,j,k,W,C) ((W * i + j)*C + k)
#define ijko2index(i,j,o,k,W,IC,OC) (((W * i + j )*OC + o)*IC + k)

int main(int argc, char * argv[]){
	int read;
	int precision;
	assert(argc == 4);
	if (strcmp(argv[3], "FP32") == 0)
		precision = 0;
	else if (strcmp(argv[3], "INT32") == 0)
		precision = 132;
	else if (strcmp(argv[3], "INT16") == 0)
		precision = 116;
	else{
		printf("WRONG\n");
		return -1;
	}
	float CON1,CON2;
	
	FILE *ptr;
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
	float *input_data;
	input_data = (float*)malloc(N * H * W * C * sizeof(float));
	read = fread(input_data, sizeof(float), N*H*W*C, ptr);
	float InputMaxValue, InputMinValue;
	InputMinValue = 0.0;
	InputMaxValue = 0.0;
	for (int i = 0; i< N*H*W*C; i++){
		if (input_data[i] > InputMaxValue) InputMaxValue = input_data[i];
		if (input_data[i] < InputMinValue) InputMinValue = input_data[i];
	}
	fclose(ptr);

	/*_______________DOWNLOAD_KERNEL_DATA___________________*/

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
	float *kernel_data;
	kernel_data = (float*)malloc(KH * KW * OC * IC * sizeof(float));
	float KernelMaxValue, KernelMinValue;
	KernelMaxValue = 0.0;
	KernelMinValue = 0.0;
	read = fread(kernel_data, sizeof(float), KH * KW * OC * IC, ptr);
	for (int i = 0; i < KH * KW * OC * IC; i++){
		if (kernel_data[i] > KernelMaxValue) KernelMaxValue = kernel_data[i];
		if (kernel_data[i] < KernelMinValue) KernelMinValue = kernel_data[i];
	}
	fclose(ptr);
	struct timeval stop, start;
	int ii,jj, PD, HP, WP;
	PD = KH/2; /*pading size*/
	HP = (H + 2 * PD); /*padded height*/
	WP = (W + 2 * PD); /*padded width*/
	float *padded; /*padded tensor*/
	padded = (float*)malloc(N * HP * WP * C * sizeof(float));
	for (int h = 0; h < N * HP * WP * C; h++){
		padded[h] = 0.0;
	}
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
	assert(HO == H); /* because padding is SAME */
	assert(HP == WP);
	int m,n,k,row;
	m = WO*HO;
	k = IC* KH * KW;
	n = OC;
	float *result;
	float overflow, MinScale, MaxScale;
	result = (float*)malloc(m * n * sizeof(float)); 
	switch (precision)
	{
		case 132:{
			printf("Case INT32\n");
			int32_t s32;
			int32_t* c32;
			c32 = (int32_t*)malloc(m * n * sizeof(int32_t*));
			int32_t *im2col32;
			int32_t *kernel2col32;
			CON1 = 2500;
			CON2 = 21410;
			printf("CON1 %f CON2 %f\n", CON1, CON2);
			/*im2col32-START*/
			im2col32 = (int32_t*)malloc(WO * HO * KH * KW * IC * sizeof(int32_t*)); 
			ix = 0;
			for( int i = kh; i < HP - kh; i++){
				jx = 0;
				for (int j = kw; j < WP - kw; j++){
					r = 0;
					for (int xx = i-kh; xx < i + kh + 1; xx++){
						for (int yy = j - kh; yy < j + kh + 1; yy++){
							for (int k = 0; k < IC; k++){
								im2col32[(KH*KW*IC)*(ix*WO + jx)  + r] = (int32_t)(padded[ijk2index(xx,yy,k,WP,IC)] * CON1);
								r++;
							}
						}
					}
					jx++;
				}
				ix++;
			}
			/*im2col8-END*/
			/*kernel2col32-START*/
			kernel2col32 = (int32_t*)malloc(KW* KH* IC * OC * sizeof(int32_t*));
			row = 0;
			for (int i = 0; i < KH; i++){
				for (int j = 0; j < KW; j++){
					for (int k = 0; k < IC; k++){
						for (int o = 0; o < OC; o++){
							kernel2col32[ij2index(o, row, KW*KH*IC)] = (int32_t)(kernel_data[ijko2index(i,j,o,k,KW,IC,OC)] * CON2);
						}
						row++;
					}
				}
			}
			/*kernel2col32-END*/
			/*MATRIX_MULTIPLICATION-START*/
			gettimeofday(&start, NULL);
			multiplyINT32(im2col32,kernel2col32, c32, m, n, k);
			gettimeofday(&stop, NULL);
			printf("Inference time is %lu microseconds\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
			/*MATRIX_MULTIPLICATION-END*/
			for (int i = 0; i < m*n; i++){
				result[i] = ((float)c32[i])/(CON1 * CON2);
			}		
			free(kernel2col32);
			free(im2col32);
			free(c32);
			break;
		}

		case 116:{
			printf("Case INT16\n");
			int16_t s16;
			int16_t *c16;
			c16 = (int16_t*)malloc(m * n * sizeof(int16_t));
			int16_t *im2col16;
			int16_t *kernel2col16;
			/*CON1*/
			CON1 = 4;
			CON2 = 350;
			printf("CON1 %f CON2 %f\n", CON1, CON2);
			im2col16 = (int16_t*)malloc(WO * HO * KH * KW * IC * sizeof(int16_t)); 
			ix = 0;
			for( int i = kh; i < HP - kh; i++){
				jx = 0;
				for (int j = kw; j < WP - kw; j++){
					r = 0;
					for (int xx = i-kh; xx < i + kh + 1; xx++){
						for (int yy = j - kh; yy < j + kh + 1; yy++){
							for (int k = 0; k < IC; k++){
								im2col16[(KH*KW*IC)*(ix*WO + jx)  + r] = (int16_t)(padded[ijk2index(xx,yy,k,WP,IC)] * CON1);
								r++;
							}
						}
					}
					jx++;
				}
				ix++;
			}
			kernel2col16 = (int16_t*)malloc(KW* KH* IC * OC * sizeof(int16_t));
			row = 0;
			for (int i = 0; i < KH; i++){
				for (int j = 0; j < KW; j++){
					for (int k = 0; k < IC; k++){
						for (int o = 0; o < OC; o++){
							kernel2col16[ij2index(o, row, KW*KH*IC)] = (int16_t)(kernel_data[ijko2index(i,j,o,k,KW,IC,OC)] * CON2);
						}
						row++;
					}
				}
			}
			gettimeofday(&start, NULL);
			multiplyINT16(im2col16, kernel2col16, c16, m,n,k);
			gettimeofday(&stop, NULL);
			printf("Inference time is %lu microseconds\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
			for (int i = 0; i < m*n; i++){
				result[i] = ((float)c16[i])/(CON1 * CON2);
			}		
			free(kernel2col16);
			free(im2col16);
			free(c16);
			break;
		}

		case 0:{
			printf("Case FP32\n");
			float s8;
			float *c8;
			c8 = (float*)malloc(m * n * sizeof(float*));
			float *im2col8;
			float *kernel2col8;
			im2col8 = (float*)malloc(WO * HO * KH * KW * IC * sizeof(float*)); 
			ix = 0;
			for( int i = kh; i < HP - kh; i++){
				jx = 0;
				for (int j = kw; j < WP - kw; j++){
					r = 0;
					for (int xx = i-kh; xx < i + kh + 1; xx++){
						for (int yy = j - kh; yy < j + kh + 1; yy++){
							for (int k = 0; k < IC; k++){
								im2col8[(KH*KW*IC)*(ix*WO + jx)  + r] = padded[ijk2index(xx,yy,k,WP,IC)];
								r++;
							}
						}
					}
					jx++;
				}
				ix++;
			}
			kernel2col8 = (float*)malloc(KW* KH* IC * OC * sizeof(float *));
			row = 0;
			for (int i = 0; i < KH; i++){
				for (int j = 0; j < KW; j++){
					for (int k = 0; k < IC; k++){
						for (int o = 0; o < OC; o++){
							kernel2col8[ij2index(o, row, KW*KH*IC)] = kernel_data[ijko2index(i,j,o,k,KW,IC,OC)];
						}
						row++;
					}
				}
			}
			gettimeofday(&start, NULL);
			multiplyFP32(im2col8,kernel2col8, c8, m, n, k);
			gettimeofday(&stop, NULL);
			printf("Inference time is %lu microseconds\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
			for (int i = 0; i < m*n; i++){
				result[i] = c8[i];
			}		
			free(kernel2col8);
			free(im2col8);
			free(c8);
			break;
		}
	
	} 
	ptr = fopen("output_tensor.bin", "w+");
	int output_size[4] = {N, HO, WO, OC};
	fwrite(output_size, sizeof(output_size), 1, ptr);
	fwrite(result, sizeof(float),N*HO*WO*OC, ptr);
	fclose(ptr);



	ptr = fopen("target_tensor.bin", "rb");
	if (ptr){
		float output_data[m*n];
		read = fread(output_size, sizeof(output_size), 1, ptr);
		read = fread(output_data, sizeof(output_data), 1, ptr);
		fclose(ptr); 
		float error;
		float max = output_data[0];
		float min = output_data[0];
		for (int i = 0; i < m*n; i++){
			error += (output_data[i] - result[i]) * (output_data[i] - result[i]);
			if (output_data[i] > max){
				max = output_data[i];
			}
			if (output_data[i] < min){
				min = output_data[i];
			}
		}
		error = sqrt(error/(m*n))/(max-min);
		printf("NRMSE is %f\n", error);
	} else {
		printf("Target tensor is not given\n");
	}
	

	free(padded);
	free(kernel_data);
	free(input_data);
	free(result);
	return 0;
}