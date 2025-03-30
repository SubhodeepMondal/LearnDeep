#ifndef NDYNAMIC_ARRAY
#define NDYNAMIC_ARRAY

#include <cstring>
#include <iostream>
#include <random>
#include <cstdarg>

/*
...............Dynamically allocated memories................

dimension(unsigned)
arr_dim(unsigned)
head(dim_node)
ptr(dim_node)
prev(dim_node)
data(template)

............................End..............................

*/

typedef struct dim_node
{
    unsigned value;
    struct dim_node *next;
} dim_node;

template <typename T>
class ndarray{
    std::string obj_name;
    dim_node *head = NULL;
    dim_node *ptr = NULL;
    dim_node *prev = NULL;
    // int type = typeFlag;
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

    // template <typename... args>
    // void addDimensions(unsigned, args...);

public:
    ndarray();

    // template <typename... Args>
    // ndarray(unsigned num, Args... args);

    ndarray(unsigned n, const unsigned *arr);

    // copy constructor.
    ndarray(const ndarray<T> &ndarray);

    ~ndarray();

    /// @brief returns unsigned array of dimension vector
    /// @tparam None
    /// @return returns unsigned array of dimension vector.
    const unsigned *getDimensions() const;

    /// @brief returns rank of the tensor.
    /// @tparam None
    /// @return returns unsigned array of dimension vector.
    unsigned getNoOfDimensions() const;

    /// @brief returns number of element for existing tensor
    /// @tparam None
    /// @return returns unsigned array of dimension vector.
    const unsigned getNoOfElem() const;

    /// @brief returns pointer to the first address of the tensor
    /// @tparam None
    /// @return returns pointer to the first address of the tensor.
    T *getData() const;

    void initData(T data);

    void initData(T *data);

    void initData(ndarray<double>incData);

    void initPartialData(unsigned index, unsigned n, T *data_source);

    void initRandData(int lower_limit, int upper_limit);

    void initPreinitilizedData(T *Data);

    void copyData(T *);

    void destroy();

    void printDimensions() const;

    void printData();

    void printLinearData();

    void printNoOfElements();

    void reshape() {}

    // template <typename first_dim, typename... Args>
    // void reshape(first_dim n, Args... args);

    void resetDimensions(unsigned, unsigned *);

    /// @brief sets name for the tensor
    /// @param str std::string type, assign this name to for the tensor
    void setObjName(std::string str);

    T operator()(unsigned);

    ndarray<T>&operator[](unsigned);

    ndarray<T>&operator=(const ndarray<T>&ndarray);
};

// #include "NDynamicArray.tpp"

#endif // NDYNAMIC_ARRAY