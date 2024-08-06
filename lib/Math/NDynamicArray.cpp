#include <cstdarg>
#include <iostream>
#include <random>

// #include "GPU_aux.h"

// template <typename T, int typeFlag>
// NDArray<T, typeFlag>::NDArray(unsigned n, ...)
// {
//     va_list valist;
//     int no_of_gpu;

//     nDim = n;
//     nElem = 1;
//     dimension = new unsigned[n];
//     isInitilized = 1;
//     va_start(valist, n);

//     for (int i = 0; i < n; i++)
//         dimension[i] = va_arg(valist, unsigned);

//     va_end(valist);

//     for (int i = 0; i < nDim; i++)
//         nElem *= dimension[i];

//     // GPU_aux<T>::getCUDADeviceCount(&no_of_gpu);
//     if (no_of_gpu && type)
//     {
//         type = 1;
//         // GPU_aux<T>::allocateGPUMemory(data, nElem);
//         // cudaMalloc((T **)&data, nElem * sizeof(T));
//     }
//     else
//         // CPU_aux<T>::allocateCPUMemory(data, nElem);
//         this->data = new T[nElem];
// }

// template <typename T, int typeFlag>
// NDArray<T, typeFlag>::NDArray(unsigned n, unsigned *arr, unsigned isInitilized)
// {

//     int no_of_gpu;
//     nDim = n;
//     nElem = 1;
//     dimension = new unsigned[n];

//     for (int i = 0; i < n; i++)
//         dimension[i] = arr[i];

//     for (int i = 0; i < nDim; i++)
//         nElem *= dimension[i];

//     this->isInitilized = isInitilized;
//     // GPU_aux<T>::getCUDADeviceCount(&no_of_gpu);

//     if (this->isInitilized)
//     {
//         if (no_of_gpu && type)
//         {
//             type = 1;
//             // GPU_aux<T>::allocateGPUMemory(data, nElem);
//             // cudaMalloc((T **)&data, nElem * sizeof(T));
//         }
//         else
//             // GPU_aux<T>::allocateCPUMemory(data, nElem);
//             this->data = new T[nElem];
//     }
//     else
//     {
//         // std::cout << "Setting data pointer to NULL\n" ;
//         this->data = NULL;
//     }
// }

// template <typename T, int typeFlag>
// NDArray<T, typeFlag>::NDArray(NDArray &ndarray)
// {
//     this->nDim = ndarray.nDim;
//     this->dimension = ndarray.dimension;
//     this->nElem = ndarray.nElem;
//     this->data = ndarray.data;
// }

template <typename T, int typeFlag>
unsigned *NDArray<T, typeFlag>::getDimensions()
{
    unsigned *ptr;
    ptr = dimension;
    return ptr;
}

template <typename T, int typeFlag>
unsigned NDArray<T, typeFlag>::getNoOfDimensions()
{
    return nDim;
}

template <typename T, int typeFlag>
unsigned NDArray<T, typeFlag>::getNoOfElem()
{
    return nElem;
}

template <typename T, int typeFlag>
T *NDArray<T, typeFlag>::getData()
{
    return data;
}

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::printDimensions()
{
    std::cout << "[ ";
    for (int i = 0; i < nDim; i++)
        std::cout << dimension[i] << ", ";
    std::cout << "]";
}

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::printData()
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

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::printLinearData()
{
    for (int i = 0; i < nElem; i++)
    {
        std::cout << data[i] << ", ";
    }
    std::cout << std::endl;
}

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::initData(T data)
{
    if (type)
    {
        T *item = new T[nElem];
        for (int i = 0; i < nElem; i++)
            item[i] = data;
        // GPU_aux<T>::cudaMemoryCopyHostToDevice(this->data,item,nElem);
        // cudaMemcpy(this->data, item, sizeof(T) * nElem, cudaMemcpyHostToDevice);
        delete[] item;
    }
    else
        for (int i = 0; i < nElem; i++)
            this->data[i] = data;
}

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::initData(T *data)
{
    if (type)
    {
        // GPU_aux<T>::cudaMemmoryCopyToDevice(this->data, data, nElem);
    }
    else
        for (int i = 0; i < nElem; i++)
            this->data[i] = data[i];
};

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::initData(NDArray<double, 1> data)
{
    if (type)
    {
        // GPU_aux<T>::cudaMemoryDeviceToDevice(this->data, data.getData(), nElem);
        // cudaMemcpy(this->data, data.getData(), sizeof(T) * nElem, cudaMemcpyDeviceToDevice);
    }
    else
    {
        // GPU_aux<T>::cudaMemoryCopyDeviceToHost(this->data, data.getData(), nElem);
        // cudaMemcpy(this->data, data.getData(), sizeof(T) * nElem, cudaMemcpyDeviceToHost);
    }
}

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::initData(NDArray<double, 0> incData)
{
    if (type)
    {
        // GPU_aux<T>::cudaMemoryCopyHostToDevice(this->data, incData.getData(), nElem);
        // cudaMemcpy(this->data, incData.getData(), sizeof(T) * nElem, cudaMemcpyHostToDevice);
    }
    else
    {
        T *ptr = incData.getData();
        for (int i = 0; i < nElem; i++)
            this->data[i] = ptr[i];
    }
}

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::initPartialData(unsigned index, unsigned n, T *data_source)
{
    int j = 0;
    if (type)
    {
        // GPU_aux<T>::cudaMemoryCopyHostToDevice(data+index, data_source,n);
        // cudaMemcpy((data + index), data_source, sizeof(T) * n, cudaMemcpyHostToDevice);
    }
    else
        for (int i = index; i < (index + n); i++)
            data[i] = data_source[j++];
}

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::initRandData(int lower_limit, int upper_limit)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution((0 + lower_limit), (1 * upper_limit));

    for (int i = 0; i < nElem; i++)
        data[i] = distribution(generator);
}

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::initPreinitilizedData(T *Data)
{
    this->data = Data;
}

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::copyData(T *data)
{
    for (int i = 0; i < nElem; i++)
        data[i] = this->data[i];
    std::cout << std::endl;
}

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::destroy()
{
    if (typeFlag)
    {
        // GPU_aux<T>::cudaMemoryFree(data);
    }
    else
        delete[] data;

    delete[] dimension;
}