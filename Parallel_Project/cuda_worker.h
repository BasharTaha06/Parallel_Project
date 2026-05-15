#ifndef CUDA_WORKER_H
#define CUDA_WORKER_H

#ifdef __cplusplus
extern "C" {
#endif

void process_matrix_cuda(int *mat);

#ifdef __cplusplus
}
#endif

#endif