#pragma ONCE
#include "CPU_aux.h"
#include "GPU_aux.h"

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
    unsigned nDim, isDimInistilized, isInitilized, no_of_gpu;
    unsigned *dimension;
    unsigned nElem;
    T *data;

    void addDimensions(unsigned w)
    {
        dim_node *ptr_new = new dim_node;
        if (!head)
        {

            head = ptr = ptr_new;

            head->value = w;
            head->next = NULL;
            nDim++;
        }
        else
        {
            ptr->next = ptr_new;
            ptr_new->value = w;
            ptr_new->next = NULL;

            ptr = ptr->next;
            nDim++;
        }
    }

    void addDimensions(unsigned *w)
    {
        isDimInistilized = 1;
        nDim = head->value;
        dimension = new unsigned[nDim];
        for (unsigned i = 0; i < nDim; i++)
        {
            dimension[i] = w[i];
        }
    }

    template <typename... args>
    void addDimensions(unsigned w, args... Args)
    {
        addDimensions(w);
        // (addDimensions(Args), ...);
        addDimensions(Args...);
    }

public:
    // NDArray(unsigned n, unsigned *arr, unsigned isInitilized = 1);

    template <typename... args>
    NDArray(args... Args)
    {
        unsigned i = 0;
        nElem = 1;
        isDimInistilized = 0;
        isInitilized = 1;
        nDim = 0;

        addDimensions(Args...);
        // (addDimensions(Args), ...);

        if (!isDimInistilized)
        {
            dimension = new unsigned[nDim];
            ptr = head;

            while (ptr)
            {
                dimension[i++] = ptr->value;
                prev = ptr;
                ptr = ptr->next;
                delete[] prev;
            }
        }

        for (i = 0; i < nDim; i++)
            nElem *= dimension[i];

        data = new T[nElem];
    }

    NDArray() {};

    // NDArray(NDArray &ndarray);

    ~NDArray() {};

    unsigned *getDimensions();

    unsigned getNoOfDimensions();

    unsigned getNoOfElem();

    T *getData();

    void printDimensions();

    void printData();

    void printLinearData();

    void initData(T data);

    void initData(T *data);

    void initData(NDArray<double, 1> data);

    void initData(NDArray<double, 0> incData);

    void initPartialData(unsigned index, unsigned n, T *data_source);

    void initRandData(int lower_limit, int upper_limit);

    void initPreinitilizedData(T *Data);

    void copyData(T *);

    void destroy();
};

#include "../lib/Math/NDynamicArray.cpp"
