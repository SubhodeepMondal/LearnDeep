#include "CPU_aux.h"
#include "CPULibrary.h"

template <typename T>
void CPU_aux<T>::allocateCPUMemory(T *data, unsigned nElement)
{
    data = new T[nElement];
}

void cpu::__mmul(double *A, double *B, double *C, unsigned a_m, unsigned a_n, unsigned b_n)
{
    double sum = 0;
    for (int i = 0; i < a_m; i++)
        for (int j = 0; j < b_n; j++)
        {
            sum = 0;
            for (int k = 0; k < a_n; k++)
                sum += A[i * a_n + k] * B[k * b_n + j];
            C[i * b_n + j] = sum;
        }
}