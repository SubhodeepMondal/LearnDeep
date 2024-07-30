#pragma ONCE
#include "CPU_aux.h"
#include "GPU_aux.h"

template <typename T, int typeFlag>
class NDArray : protected CPU_aux<T>,
                protected GPU_aux<T>
{
    int type = typeFlag;
    unsigned nDim, isInitilized;
    unsigned *dimension;
    unsigned nElem;
    T *data;

public:
    NDArray(unsigned, ...);

    NDArray(unsigned n, unsigned *arr, unsigned isInitilized = 1);

    NDArray(NDArray &ndarray);

    ~NDArray() {};

    NDArray() {};

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
