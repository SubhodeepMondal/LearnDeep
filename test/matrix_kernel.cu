#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <random.h>

__global__ cudamatmul(float *a, float *b, float *c, unsigned m, unsigned n, unsigned k)
{
    unsigned indx_a, indx_b, indx_c, indx, indy;
    indx = threadIdx.x + blockIdx.x + blockDim.x;
    indy = threadIdx.y + blockIdx.y + blockDim.y;
    float sum = 0.0f;

    if (indx < m && indy < n)
    {
        for (int i = 0; i < k; i++)
            sum += a[i + indy * m] * b[i * n + indx];
        c[indx + indy * m] = sum;
    }
}

void main()
{
    unsigned m_dim, n_dim, k_dim;
    m_dim = 1 << 10;
    n_dim = 1 << 10;
    k_dim = 1 << 8;

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float *)malloc(k_dim * m_dim * sizeof(float));
    h_B = (float *)malloc(n_dim * k_dim * sizeof(float));
    h_C = (float *)malloc(n_dim * m_dim * sizeof(float));

    cudaMalloc((float **)&d_A, m_dim * k_dim * sizeof(float));
    cudaMalloc((float **)&d_B, m_dim * k_dim * sizeof(float));
    cudaMalloc((float **)&d_C, m_dim * k_dim * sizeof(float));

    for (unsigned i = 0; i < m_dim * k_dim; i++)
    {
        h_A[i] = rand() / (float)10000;
        h_B[i] = rand() / (float)10000;
    }

    cudaMemcpy(d_A, h_A, k_dim * m_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k_dim * m_dim * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid, block;
    block.x = 32;
    block.y = 32;
    grid.x = (m_dim) / block.x;
    grid.y = (n_dim) / block.y;

    cudamatmul<<<grid, block>>>(d_A, d_B, d_C, m_dim, n_dim, k_dim);

    cudaMemcpy(h_C, d_C, m_dim * n_dim * sizeof(float), cudaMemcpyDeviceToHost);

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}