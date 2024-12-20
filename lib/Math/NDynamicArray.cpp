#include <iostream>
#include <random>
#include <cstdarg>

template <typename T, int typeFlag>
NDArray<T, typeFlag>::NDArray()
{
    dim_iterator = 0;
    nDim = 0;
    no_of_gpu = 0;
    dimension = nullptr;
    arr_dim = nullptr;
    data = nullptr;

    // std::cout << "default constructor data: " << data << " dim ptr: " << dimension << " arr ptr: " << arr_dim << "\n";
}

template <typename T, int typeFlag>
NDArray<T, typeFlag>::NDArray(const NDArray<T, typeFlag> &ndarray)
{
    if (isInitilized)
        destroy();
    isInitilized = true;
    isDimInitilized = true;
    this->nDim = ndarray.getNoOfDimensions();

    dimension = new unsigned[nDim];
    arr_dim = new unsigned[nDim];

    unsigned *temp_arr = new unsigned[nDim];

    for (unsigned i = 0; i < nDim; i++)
        temp_arr[i] = 0;

    std::memcpy(dimension, ndarray.getDimensions(), nDim * sizeof(unsigned));
    std::memcpy(arr_dim, temp_arr, nDim * sizeof(unsigned));

    nElem = ndarray.getNoOfElem();
    data = new T[nElem];

    this->initData(ndarray.getData());
    // std::cout << "copy constructor data: " << data << " dim ptr: " << dimension << " arr ptr: " << arr_dim << "\n";
}

template <typename T, int typeFlag>
template <typename... Args>
NDArray<T, typeFlag>::NDArray(unsigned num, Args... args)
{
    unsigned i;
    nElem = 1;
    isDimInitilized = false;
    isInitilized = true;
    nDim = 0;

    dim_node *ptr_new = new dim_node;

    head = ptr = ptr_new;
    head->value = num;
    head->next = NULL;
    nDim++;

    addDimensions(args...);
    i = nDim - 1;

    if (!isDimInitilized)
    {
        isDimInitilized = true;
        dimension = new unsigned[nDim];
        arr_dim = new unsigned[nDim];
        ptr = head;
        while (ptr)
        {
            dimension[i--] = ptr->value;
            prev = ptr;
            ptr = ptr->next;
            delete prev;
        }
    }

    for (i = 0; i < nDim; i++)
        nElem *= dimension[i];

    data = new T[nElem];
}

template <typename T, int typeFlag>
NDArray<T, typeFlag>::~NDArray()
{
    // printDimensions();
    // std::cout << "Destroyer " << isInitilized << ", " << isDimInitilized << "\n";
    // std::cout << "data: " << data << " dim ptr: " << dimension << " arr ptr: " << arr_dim << "\n";

    // std::cout << "destroyer destroying data for object:" << obj_name << " " << data << " " << dimension << " " << arr_dim <<  "!\n";
    if (data)
    {
        delete[] data;
        data = nullptr;
        // std::cout << "freed data memory \n ";
    }

    if (dimension)
    {
        delete[] dimension;
        dimension = nullptr;
        // std::cout << "freed dimension memory \n ";
    }

    if (arr_dim)
    {
        delete[] arr_dim;
        arr_dim = nullptr;
        // std::cout << "freed arr_dim memory \n ";
    }
    // std::cout << "destroyed!\n";
}

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
template <typename first_dim, typename... Args>
void NDArray<T, typeFlag>::reshape(first_dim n, Args... args)
{
    if (dim_iterator == nDim - 1)
    {
        arr_dim[dim_iterator] = n;
        dim_iterator = 0;
        unsigned product_a, product_b;

        product_a = product_b = 1;

        for (unsigned i = 0; i < nDim; i++)
        {
            product_a *= dimension[i];
            product_b *= arr_dim[i];
        }
        if (product_a == product_b)
        {
            for (unsigned i = 0; i < nDim; i++)
                dimension[i] = arr_dim[i];
        }
        else
        {
            std::cout << "reshape is not possible!\n";
        }

        // delete[] arr_dim;
    }
    else
    {

        arr_dim[dim_iterator++] = n;

        reshape(args...);
    }
    // delete [] arr_dim;
}

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::destroy()
{
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
    // std::cout << "destroyed!\n";
}

template <typename T, int typeFlag>
T NDArray<T, typeFlag>::operator()(unsigned n)
{
    unsigned loc = n;
    if (dim_iterator == nDim - 1)
    {
        for (int i = 0; i < dim_iterator; i++)
        {
            loc += arr_dim[i] * dimension[i];
        }
        dim_iterator = 0;

        return data[loc];
    }
    else
    {
        std::cout << "Invalid indexing.\n";
        return (T)0;
    }
}

template <typename T, int typeFlag>
NDArray<T, typeFlag> &NDArray<T, typeFlag>::operator[](unsigned n)
{
    arr_dim[dim_iterator++] = n;
    return *this;
}

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::setObjName(std::string str) { obj_name = str; }

template <typename T, int typeFlag>
void NDArray<T, typeFlag>::printNoOfElements() { std::cout << nElem << "\n"; }

template <typename T, int typeFlag>
NDArray<T, typeFlag> &NDArray<T, typeFlag>::operator=(const NDArray<T, typeFlag> &ndarray)
{
    // std::cout << "inside assignment operator" << this << " " << &ndarray << "\n";
    if (this == &ndarray)
        return *this;
    if (isInitilized)
        destroy();
    isInitilized = true;
    isDimInitilized = true;

    this->nDim = ndarray.getNoOfDimensions();
    dimension = new unsigned[nDim];
    arr_dim = new unsigned[nDim];
    unsigned *temp_arr = new unsigned[nDim];
    for (unsigned i = 0; i < nDim; i++)
        temp_arr[i] = 0;
    std::memcpy(dimension, ndarray.getDimensions(), nDim * sizeof(unsigned));
    std::memcpy(arr_dim, temp_arr, nDim * sizeof(unsigned));
    nElem = ndarray.getNoOfElem();
    data = new T[nElem];
    if (ndarray.isInitilized)
        this->initData(ndarray.getData());

    // std::cout << "assignment operator data: " << data << " dim ptr: " << dimension << " arr ptr: " << arr_dim << "\n";
    return *this;
}
