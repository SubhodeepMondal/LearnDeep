#include <iostream>
#include <random>
#include <cstdarg>

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::addDimensions(unsigned w)
{
    dim_node *ptr_new = new dim_node;

    ptr->next = ptr_new;
    ptr_new->value = w;
    ptr_new->next = NULL;
    ptr = ptr->next;
    nDim++;
}

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::addDimensions(const unsigned *w)
{
    isDimInitilized = true;
    nDim = head->value;
    dimension = new unsigned[nDim];
    arr_dim = new unsigned[nDim];
    for (unsigned i = 0; i < nDim; i++)
    {
        dimension[i] = w[i];
    }

    delete head;
}

template <typename T, int typeFlag>
template <typename... args>
void NDArray<T, typeFlag>::addDimensions(unsigned w, args... Args)
{
    addDimensions(w);
    addDimensions(Args...);
}

template <typename T, int typeFlag>
const unsigned *NDArray<T, typeFlag>::getDimensions() const { return dimension; }

template <typename T, int typeFlag>
unsigned NDArray<T, typeFlag>::getNoOfDimensions() const { return nDim; }

template <typename T, int typeFlag>
const unsigned NDArray<T, typeFlag>::getNoOfElem() const { return nElem; }

template <typename T, int typeFlag>
T *NDArray<T, typeFlag>::getData() const { return data; }

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::printDimensions() const
{
    std::cout << "[ ";
    for (int i = nDim - 1; i >= 0; i--)
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
        std::memcpy(this->data, data, nElem * sizeof(T));
};

// template <typename T, int typeFlag>
// void NDArray<T, typeFlag>::initData(NDArray<double, 1> data)
// {
//     if (type)
//     {
//         // GPU_aux<T>::cudaMemoryDeviceToDevice(this->data, data.getData(), nElem);
//         // cudaMemcpy(this->data, data.getData(), sizeof(T) * nElem, cudaMemcpyDeviceToDevice);
//     }
//     else
//     {
//         // GPU_aux<T>::cudaMemoryCopyDeviceToHost(this->data, data.getData(), nElem);
//         // cudaMemcpy(this->data, data.getData(), sizeof(T) * nElem, cudaMemcpyDeviceToHost);
//     }
// }

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::initData(NDArray<double, 0> incData)
{

    nDim = incData.getNoOfDimensions();
    nElem = 1;
    dimension = new unsigned[nDim];
    for (unsigned i = 0; i < nDim; i++)
    {
        dimension[i] = incData.getDimensions()[i];
        nElem *= dimension[i];
    }

    if (type)
    {
        // GPU_aux<T>::cudaMemoryCopyHostToDevice(this->data, incData.getData(), nElem);
        // cudaMemcpy(this->data, incData.getData(), sizeof(T) * nElem, cudaMemcpyHostToDevice);
    }
    else
    {
        data = new T[nElem];
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
void NDArray<T, typeFlag>::resetDimensions(unsigned n, unsigned *arr)
{

    nDim = n;
    nElem = 1;
    dimension = new unsigned[nDim];
    for (unsigned i = 0; i < nDim; i++)
    {
        dimension[i] = arr[i];
        nElem *= dimension[i];
    }

    data = new T[nElem];
}

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::destroy()
{
    // std::cout << "destroy data:\n" << data << " dim ptr: " << dimension << " arr ptr: " << arr_dim << "\n";
    if (data)
    {
        delete[] data;
        data = nullptr;
        // std::cout << "freed data memory in destroy \n ";
    }

    if (dimension)
    {
        delete[] dimension;
        dimension = nullptr;
        // std::cout << "freed dimension memory in destroy\n ";
    }
    if (arr_dim)
    {
        delete[] arr_dim;
        arr_dim = nullptr;
        // std::cout << "freed arr_dim memory in destroy\n ";
    }
}