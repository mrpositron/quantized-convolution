#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "mulINT32.h"

#define tnum 16


int NN,MM,KK;
int32_t *AA;
int32_t *BB;
int32_t *CC;
typedef struct arguments{
    int i0;
    int i1;
}arguments;

void* run_thread_INT32(void* args){
    arguments* argument = (arguments*)args;
    int i0 = argument->i0;
    int i1 = argument->i1;
    for(int i = i0; i < i1 + 1; i++){
        for (int j = 0; j < NN; j++) {
            int32_t sum = 0;
            for (int k = 0; k < KK/8; k++){
            	int indd1 = i*KK + 8*k;
            	int indd2 = j*KK + 8*k;
            	__m256i m1 = _mm256_set_epi32(AA[indd1], AA[indd1+1], AA[indd1+2], AA[indd1+3], AA[indd1+4], AA[indd1+5], AA[indd1+6], AA[indd1+7]);
    			__m256i m2 = _mm256_set_epi32(BB[indd2], BB[indd2+1], BB[indd2+2], BB[indd2+3], BB[indd2+4], BB[indd2+5], BB[indd2+6], BB[indd2+7]);
    			__m256i result = _mm256_mullo_epi32(m1,m2);
    			int32_t* res = (int32_t*)&result;
    			sum += res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];
            }
            CC[i*NN + j] = sum;
        }
    }
    pthread_exit(NULL);
}

void multiplyINT32(int32_t *A, int32_t *B, int32_t *C, int M, int N, int K) {
    NN = N;
    MM = M;
    KK = K;
    AA = A;
    BB = B;
    CC = C;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t thread_pool[tnum];
    arguments args_pool[tnum];
    int n_split = tnum;
    int n_work = M/tnum;

    for (int i = 0; i < n_split; i++) {
        arguments args;
        args.i0 = i * n_work;
        args.i1 = args.i0 + n_work;
        args_pool[i] = args;
        pthread_create(&thread_pool[i], &attr, run_thread_INT32, (void*) &args_pool[i]);
    }

    for(int i = 0; i < n_split; i++){
        pthread_join(thread_pool[i], NULL);
    }
    pthread_attr_destroy(&attr);
}