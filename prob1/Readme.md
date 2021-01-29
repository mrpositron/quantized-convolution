# Readme

(1) Implementation of the convolution in C was done using im2col[1] technique. In this technique, input tensor and kernel tensor are transformed to two 2D matrices. Afterwards, single dot product is performed.

The command to run the code is as follows:

```./convolution input_tensor.bin kernel_tensor.bin```

The result of the convolution operation will be saved in output_tensot.bin file.

(2) The running time for transforming matrices is much smaller that the actual multiplication. Thus, only time spent on dot product was measured.

| The name of folders that contains input tensor and kernel tensor | Inference time (in milliseconds) |
|-------------------------------------------------------------------|----------------------------------|
| sample 1                                                            | 136.000                          |
| sample 2                                                            | 369.055                          |
| sample 3                                                            | 706.000                          |
