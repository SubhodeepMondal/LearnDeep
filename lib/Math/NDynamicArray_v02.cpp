#include <cstdarg>
#include <iostream>
#include <random>

// #include "GPU_aux.h"
#include "CPU_aux.h"
#include "NDynamicArray.h"

template<>
NDArray<double, 0>::NDArray(unsigned n, ...)
{
    va_list valist;
    int no_of_gpu;

    nDim = n;
    nElem = 1;
    dimension = new unsigned[n];
    isInitilized = 1;
    va_start(valist, n);

    for (int i = 0; i < n; i++)
        dimension[i] = va_arg(valist, unsigned);

    va_end(valist);

    for (int i = 0; i < nDim; i++)
        nElem *= dimension[i];

    // GPU_aux<double>::getCUDADeviceCount(&no_of_gpu);
    if (no_of_gpu && type)
    {
        type = 1;
        // GPU_aux<double>::allocateGPUMemory(data, nElem);
        // cudaMalloc((double **)&data, nElem * sizeof(double));
    }
    else
        // CPU_aux<double>::allocateCPUMemory(data, nElem);
        this->data = new double[nElem];
}

template<>
NDArray<double, 0>::NDArray(unsigned n, unsigned *arr, unsigned isInitilized)
{

    int no_of_gpu;
    nDim = n;
    nElem = 1;
    dimension = new unsigned[n];

    for (int i = 0; i < n; i++)
        dimension[i] = arr[i];

    for (int i = 0; i < nDim; i++)
        nElem *= dimension[i];

    this->isInitilized = isInitilized;
    // GPU_aux<double>::getCUDADeviceCount(&no_of_gpu);

    if (this->isInitilized)
    {
        if (no_of_gpu && type)
        {
            type = 1;
            // GPU_aux<double>::allocateGPUMemory(data, nElem);
            // cudaMalloc((double **)&data, nElem * sizeof(double));
        }
        else
            // GPU_aux<double>::allocateCPUMemory(data, nElem);
            this->data = new double[nElem];
    }
    else
    {
        // std::cout << "Setting data pointer to NULL\n" ;
        this->data = NULL;
    }
}

template<>
NDArray<double, 0>::NDArray(NDArray &ndarray)
{
    this->nDim = ndarray.nDim;
    this->dimension = ndarray.dimension;
    this->nElem = ndarray.nElem;
    this->data = ndarray.data;
}

template<>
unsigned *NDArray<double, 0>::getDimensions()
{
    unsigned *ptr;
    ptr = dimension;
    return ptr;
}

template<>
unsigned NDArray<double, 0>::getNoOfDimensions()
{
    return nDim;
}

template<>
unsigned NDArray<double, 0>::getNoOfElem()
{
    return nElem;
}

template<>
double *NDArray<double, 0>::getData()
{
    return data;
}

template<>
void NDArray<double, 0>::printDimensions()
{
    std::cout << "[ ";
    for (int i = 0; i < nDim; i++)
        std::cout << dimension[i] << ", ";
    std::cout << "]";
}

template<>
void NDArray<double, 0>::printData()
{
    int Elem;

    int *dim;
    dim = new int[nDim];
    for (int i = 0; i < nDim; i++)
        dim[i] = dimension[i];

    for (int i = 0; i < nElem; i++)
    {
        if (dim[0] == 1)
            std::cout << "[";

        Elem = 1;
        for (int j = 0; j < nDim; j++)
        {
            Elem *= dim[j];
            if ((i + 1) % Elem == 1)
                std::cout << "[";
        }

        std::cout << "\t";

        if (type)
        {
            // printCUDAElement(data + 1);
            // gpu::print<<<1, 1>>>(data + i);
            // cudaDeviceSynchronize();
        }
        else
        {
            std::cout.precision(6);
            std::cout.setf(std::ios::showpoint);
            std::cout << data[i];
        }

        if ((i + 1) % dim[0] != 0)
            std::cout << ",";

        Elem = 1;
        for (int j = 0; j < nDim; j++)
        {
            Elem *= dim[j];
            if ((i + 1) % Elem == 0)
            {
                if (j == 0)
                    std::cout << "\t";
                std::cout << "]";
            }
        }

        if ((i + 1) % dim[0] == 0)
            std::cout << std::endl;
    }
    // std::cout << std::endl;
    free(dim);
}

template<>
void NDArray<double, 0>::printLinearData()
{
    for (int i = 0; i < nElem; i++)
    {
        std::cout << data[i] << ", ";
    }
    std::cout << std::endl;
}

template<>
void NDArray<double, 0>::initData(double data)
{
    if (type)
    {
        double *item = new double[nElem];
        for (int i = 0; i < nElem; i++)
            item[i] = data;
        // GPU_aux<double>::cudaMemoryCopyHostToDevice(this->data,item,nElem);
        // cudaMemcpy(this->data, item, sizeof(double) * nElem, cudaMemcpyHostToDevice);
        delete[] item;
    }
    else
        for (int i = 0; i < nElem; i++)
            this->data[i] = data;
}

template<>
void NDArray<double, 0>::initData(double *data)
{
    if (type)
    {
        // GPU_aux<double>::cudaMemmoryCopyToDevice(this->data, data, nElem);
    }
    else
        for (int i = 0; i < nElem; i++)
            this->data[i] = data[i];
};

template<>
void NDArray<double, 0>::initData(NDArray<double, 1> data)
{
    if (type)
    {
        // GPU_aux<double>::cudaMemoryDeviceToDevice(this->data, data.getData(), nElem);
        // cudaMemcpy(this->data, data.getData(), sizeof(double) * nElem, cudaMemcpyDeviceToDevice);
    }
    else
    {
        // GPU_aux<double>::cudaMemoryCopyDeviceToHost(this->data, data.getData(), nElem);
        // cudaMemcpy(this->data, data.getData(), sizeof(double) * nElem, cudaMemcpyDeviceToHost);
    }
}

template<>
void NDArray<double, 0>::initData(NDArray<double, 0> incData)
{
    if (type)
    {
        // GPU_aux<double>::cudaMemoryCopyHostToDevice(this->data, incData.getData(), nElem);
        // cudaMemcpy(this->data, incData.getData(), sizeof(double) * nElem, cudaMemcpyHostToDevice);
    }
    else
    {
        double *ptr = incData.getData();
        for (int i = 0; i < nElem; i++)
            this->data[i] = ptr[i];
    }
}

template<>
void NDArray<double, 0>::initPartialData(unsigned index, unsigned n, double *data_source)
{
    int j = 0;
    if (type)
    {
        // GPU_aux<double>::cudaMemoryCopyHostToDevice(data+index, data_source,n);
        // cudaMemcpy((data + index), data_source, sizeof(double) * n, cudaMemcpyHostToDevice);
    }
    else
        for (int i = index; i < (index + n); i++)
            data[i] = data_source[j++];
}

template<>
void NDArray<double, 0>::initRandData(int lower_limit, int upper_limit)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution((0 + lower_limit), (1 * upper_limit));

    for (int i = 0; i < nElem; i++)
        data[i] = distribution(generator);
}

template<>
void NDArray<double, 0>::initPreinitilizedData(double *Data)
{
    this->data = Data;
}

template<>
void NDArray<double, 0>::copyData(double *data)
{
    for (int i = 0; i < nElem; i++)
        data[i] = this->data[i];
    std::cout << std::endl;
}

template<>
void NDArray<double, 0>::destroy()
{
    if (0)
    {
        // GPU_aux<double>::cudaMemoryFree(data);
    }
    else
        delete[] data;

    delete[] dimension;
}