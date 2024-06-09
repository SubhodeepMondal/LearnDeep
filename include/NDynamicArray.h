#pragma ONCE

#include <cstdarg>
#include <iostream>
#include <random>
#include "GPULibrary.h"

template <class T, int typeFlag>
class NDArray
{
    int type = typeFlag;
    unsigned nDim, isInitilized;
    unsigned *dimension;
    unsigned nElem;
    T *data;

public:
    NDArray(unsigned n, ...)
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

        cudaGetDeviceCount(&no_of_gpu);
        if (no_of_gpu && type)
        {
            type = 1;
            cudaMalloc((T **)&data, nElem * sizeof(T));
        }
        else
            this->data = new T[nElem];
    }

    NDArray(unsigned n, unsigned *arr, unsigned isInitilized = 1)
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
        cudaGetDeviceCount(&no_of_gpu);

        if (this->isInitilized)
        {
            if (no_of_gpu && type)
            {
                type = 1;
                cudaMalloc((T **)&data, nElem * sizeof(T));
            }
            else
                this->data = new T[nElem];
        }
        else
        {
            // std::cout << "Setting data pointer to NULL\n" ;
            this->data = NULL;
        }
    }

    NDArray(NDArray &ndarray)
    {
        this->nDim = ndarray.nDim;
        this->dimension = ndarray.dimension;
        this->nElem = ndarray.nElem;
        this->data = ndarray.data;
    }

    ~NDArray(){};

    NDArray(){};

    unsigned *getDimensions()
    {
        unsigned *ptr;
        ptr = dimension;
        return ptr;
    }

    unsigned getNoOfDimensions()
    {
        return nDim;
    }

    unsigned getNoOfElem()
    {
        return nElem;
    }

    T *getData()
    {
        return data;
    }

    void printDimensions()
    {
        std::cout << "[ ";
        for (int i = 0; i < nDim; i++)
            std::cout << dimension[i] << ", ";
        std::cout << "]";
    }

    void printData()
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
                gpu::print<<<1, 1>>>(data + i);
                cudaDeviceSynchronize();
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

    void printLinearData()
    {
        for (int i = 0; i < nElem; i++)
        {
            std::cout << data[i] << ", ";
        }
        std::cout << std::endl;
    }

    void initData(T data)
    {
        if (type)
        {
            T *item = new T[nElem];
            for (int i = 0; i < nElem; i++)
                item[i] = data;
            cudaMemcpy(this->data, item, sizeof(T) * nElem, cudaMemcpyHostToDevice);
            delete[] item;
        }
        else
            for (int i = 0; i < nElem; i++)
                this->data[i] = data;
    }

    void initData(T *data)
    {
        if (type)
            cudaMemcpy(this->data, data, sizeof(T) * nElem, cudaMemcpyHostToDevice);
        else
            for (int i = 0; i < nElem; i++)
                this->data[i] = data[i];
    };

    void initData(NDArray<double, 1> data)
    {
        if (type)
            cudaMemcpy(this->data, data.getData(), sizeof(T) * nElem, cudaMemcpyDeviceToDevice);
        else
            cudaMemcpy(this->data, data.getData(), sizeof(T) * nElem, cudaMemcpyDeviceToHost);
    }

    void initData(NDArray<double, 0> incData)
    {
        if (type)
            cudaMemcpy(this->data, incData.getData(), sizeof(T) * nElem, cudaMemcpyHostToDevice);
        else
        {
            T *ptr = incData.getData();
            for (int i = 0; i < nElem; i++)
                this->data[i] = ptr[i];
        }
    }

    void initPartialData(unsigned index, unsigned n, T *data_source)
    {
        int j = 0;
        if (type)
            cudaMemcpy((data + index), data_source, sizeof(T) * n, cudaMemcpyHostToDevice);
        else
            for (int i = index; i < (index + n); i++)
                data[i] = data_source[j++];
    }

    void initRandData(int lower_limit, int upper_limit)
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution((0 + lower_limit), (1 * upper_limit));

        for (int i = 0; i < nElem; i++)
            data[i] = distribution(generator);
    }

    void initPreinitilizedData(T *Data)
    {
        this->data = Data;
    }

    void copyData(T *data)
    {
        for (int i = 0; i < nElem; i++)
            data[i] = this->data[i];
        std::cout << std::endl;
    }

    void destroy()
    {
        if (typeFlag)
            cudaFree(data);
        else
            delete[] data;

        delete[] dimension;
    }
};
