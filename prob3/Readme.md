# Readme

1. Implementation of the quantized convolution using pthreads and AVX instructions using INT16 and INT32.

The command to run the code is as follows:

```
./convolution input_tensor.bin kernel_tensor.bin [FP32/INT32/INT16]
```

The code will save the result in output_tensor.bin. It will also compute NRMSE. In the code you
should specify the name of the target binary file (from Prob 1) to compute NRMSE.

2. Same constants as in Prob 2 were used. Thus, the NRMSE did not change. There is no accuracy loss for FP32 obviously.


|          | Inference time (us) |
|----------|---------------------|
| sample 1 |                     |
| INT32    | 47165               |
| INT16    | 38887               |
| FP32     | 50000               |
| sample 2 |                     |
| INT32    | 47592               |
| INT16    | 45000               |
| FP32     | 60000               |
| sample 3 |                     |
| INT32    | 68000               |
| INT16    | 62000               |
| FP32     | 75000               |
