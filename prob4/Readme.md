# Readme

1. Convolution implented using CUDA. 

The command to run the code is as follows:

```
./convolution input_tensor.bin kernel_tensor.bin
```

2. The performance of CUDA implementation.

|          | Inference time on CUDA \* (us) | Overall inference time (us) |
|----------|-------------------------------|-----------------------------|
| sample 1 | 7455                          | 165929                      |
| sample 2 | 12034                         | 176000                      |
| sample 3 | 5549                          | 178000                      |

\* - not counting costly data transferring operations
