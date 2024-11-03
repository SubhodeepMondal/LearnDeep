#include <iostream>
#include "CPU_aux.h"
#include "CPULibrary.h"
#include <omp.h>

template <typename T>
void CPU_aux<T>::allocateCPUMemory(T *data, unsigned nElement)
{
    data = new T[nElement];
}

void cpu::__mmul(double *A, double *B, double *C, unsigned x, unsigned y, unsigned z)
{
    // x output row size
    // y k row
    // z output column size
    omp_set_num_threads(12);
    std::cout << omp_get_max_threads() << "\n";
#pragma omp parallel
    {
#pragma omp for
        for (int k = 0; k < z; k++)

            for (int j = 0; j < y; j++)
            {
                for (int i = 0; i < x; i++)
                {
                    C[i + k * x] += A[j + k * y] * B[i + j * x];
                }
            }
    }
}

void cpu::__mmulconventional(double *A, double *B, double *C, unsigned x, unsigned y, unsigned z)
{

    double sum;
    // x output row size
    // y k row
    // z output column size
    omp_set_num_threads(12);
#pragma omp parallel
    {
#pragma omp for private(sum)
        for (int j = 0; j < z; j++)
            for (int i = 0; i < x; i++)
            {
                sum = 0;
                for (int k = 0; k < y; k++)
                    sum += A[k + j * y] * B[i + k * x];
                C[i + j * x] = sum;
            }
    }
}

void cpu::__mscalermul(double *A, double B, double *C, unsigned x, unsigned y)
{
    for (unsigned j = 0; j < y; j++)
        for (unsigned i = 0; i < x; i++)
            C[i + j * x] = B * A[i + j * x];
}

void cpu::__madd(double *inp_a, double *inp_b, double *out, unsigned x, unsigned y)
{
    for (int j = 0; j < y; j++)
        for (int i = 0; i < x; i++)
            out[i + j * x] = inp_a[i + j * x] + inp_b[i + j * x];
}

void cpu::__msub(double *inp_a, double *inp_b, double *out, unsigned x, unsigned y)
{
    for (int i = 0; i < y; i++)
        for (int j = 0; j < x; j++)
            out[j + i * x] = inp_a[j + i * x] - inp_b[j + i * x];
}

void cpu::__mrollingsum(double *inp, double *output, unsigned axis, unsigned x, unsigned y, unsigned z)
{
    unsigned i, j, k, sum = 0;

    switch (axis)
    {
    case 0:
    {
        for (j = 0; j < z; j++)
            for (i = 0; i < y; i++)
            {
                sum = 0;
                for (k = 0; k < x; k++)
                    sum += inp[k + i * x + j * x * y];
                output[i + j * x] = sum;
            }
        break;
    }
    default:
        break;
    }
}

void cpu::__mtranspose(double *A, double *B, unsigned x, unsigned y)
{
    double **temp;

    temp = new double *[y];
    for (int j = 0; j < y; j++)
    {
        temp[j] = new double[x];
        for (int i = 0; i < x; i++)

            temp[j][i] = A[j + i * y];
    }

    for (int j = 0; j < y; j++)
    {
        for (int i = 0; i < x; i++)
            B[i + j * x] = temp[j][i];

        delete[] temp[j];
    }

    delete[] temp;
}