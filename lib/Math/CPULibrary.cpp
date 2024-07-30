#include "CPU_aux.h"

template <typename T>
void CPU_aux<T>::allocateCPUMemory(T *data, unsigned nElement)
{
    data = new T[nElement];
}