#include <stdio.h>

__global__ void multiplyMatrixKernel(int *mat, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        mat[idx] = mat[idx] * 10; 
    }
}

extern "C" void process_matrix_cuda(int *mat) {
    int size = 50 * 50;
    int *d_mat; 
    
    cudaMalloc((void**)&d_mat, size * sizeof(int));
    
    cudaMemcpy(d_mat, mat, size * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    multiplyMatrixKernel<<<blocksPerGrid, threadsPerBlock>>>(d_mat, size);
    
    cudaMemcpy(mat, d_mat, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_mat);
}