#pragma ONCE

template <typename T>
class GPU_aux
{
protected:
    void allocateGPUMemory(T *, unsigned);

    void getCUDADeviceCount(unsigned *);

    void printCUDAElement(T *);

    void cudaMemoryCopyHostToDevice(T *, T *, unsigned);

    void cudaMemoryCopyDeviceToDevice(T *, T *, unsigned);

    void cudaMemoryCopyDeviceToHost(T *, T *, unsigned);

    void cudaMemoryFree(T*);
};