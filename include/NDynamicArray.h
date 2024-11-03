#pragma ONCE
#include "CPU_aux.h"
#include "GPU_aux.h"
#include <cstring>
/*
...............Dynamically allocated memories................

dimension(unsigned)
arr_dim(unsigned)
head(dim_node)
ptr(dim_node)
prev(dim_node)
data(template)

*/
typedef struct dim_node
{
    unsigned value;
    struct dim_node *next;
} dim_node;

template <typename T, int typeFlag>
class NDArray : protected CPU_aux<T>,
                protected GPU_aux<T>
{

    dim_node *head = NULL;
    dim_node *ptr = NULL;
    dim_node *prev = NULL;
    int type = typeFlag;
    unsigned nDim, nElem;
    unsigned no_of_gpu;
    unsigned *dimension, *arr_dim;
    unsigned dim_iterator;
    bool isDimInitilized = false;
    bool isInitilized = false;
    T *data;

    void addDimensions() {};

    void addDimensions(unsigned);

    void addDimensions(const unsigned *);

    template <typename... args>
    void addDimensions(unsigned, args...);

public:
    // NDArray(unsigned n, unsigned *arr, unsigned isInitilized = 1);

    NDArray()
    {
        dim_iterator = 0;
        nDim = 0;
        no_of_gpu = 0;
        dimension = nullptr;
        arr_dim = nullptr;
        data = nullptr;

        // std::cout << "default constructor data: " << data << " dim ptr: " << dimension << " arr ptr: " << arr_dim << "\n";
    }

    // copy constructor.
    NDArray(const NDArray<T, typeFlag> &ndarray)
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

    NDArray<T, typeFlag>& operator=(const NDArray<T, typeFlag>& ndarray)
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
        this->initData(ndarray.getData());

        // std::cout << "assignment operator data: " << data << " dim ptr: " << dimension << " arr ptr: " << arr_dim << "\n";
        return *this;
    }

    template <typename... Args>
    NDArray(unsigned num, Args... args)
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
        // std::cout << "overloaded constructor data: " << data << " dim ptr: " << dimension << " arr ptr: " << arr_dim << "\n";
    }

    ~NDArray()
    {
        // printDimensions();
        // std::cout << "Destroyer " << isInitilized << ", " << isDimInitilized << "\n";
        // std::cout << "data: " << data << " dim ptr: " << dimension << " arr ptr: " << arr_dim << "\n";
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
    }

    const unsigned *getDimensions() const;

    unsigned getNoOfDimensions() const;

    const unsigned getNoOfElem() const;

    T *getData() const;

    void printDimensions() const;

    void printData();

    void printLinearData();

    void initData(T data);

    void initData(T *data);

    // void initData(NDArray<double, 1> data);

    void initData(NDArray<double, 0> incData);

    void initPartialData(unsigned index, unsigned n, T *data_source);

    void initRandData(int lower_limit, int upper_limit);

    void initPreinitilizedData(T *Data);

    void copyData(T *);

    void destroy();

    void reshape()
    {
    }

    template <typename first_dim, typename... Args>
    void reshape(first_dim n, Args... args)
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

    void resetDimensions(unsigned, unsigned *);

    T operator()(unsigned n)
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

    NDArray<T, typeFlag> &operator[](unsigned n)
    {
        arr_dim[dim_iterator++] = n;
        return *this;
    }
};

#include "../lib/Math/NDynamicArray.cpp"
