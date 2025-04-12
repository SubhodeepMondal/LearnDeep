#include "NDynamicArray.h"

template <typename T> ndarray<T>::ndarray() {
  dim_iterator = 0;
  nDim = 0;
  no_of_gpu = 0;
  dimension = nullptr;
  arr_dim = nullptr;
  data = nullptr;

  // std::cout << "default constructor data: " << data << " dim ptr: " <<
  // dimension << " arr ptr: " << arr_dim << "\n";
}

template <typename T> ndarray<T>::ndarray(const ndarray<T> &ndarray) {
  if (isInitilized)
    destroy();
  isInitilized = true;
  isDimInitilized = true;
  this->nDim = ndarray.getNoOfDimensions();

  dimension = new unsigned[nDim];
  arr_dim = new unsigned[nDim];

  std::memcpy(dimension, ndarray.getDimensions(), nDim * sizeof(unsigned));
  std::memset(arr_dim, 0, nDim * sizeof(unsigned));

  nElem = ndarray.getNoOfElem();
  data = new T[nElem];
  if (ndarray.isInitilized)
    this->initData(ndarray.getData());
}

// template <typename T>
// template <typename... Args>
// ndarray<T>::ndarray(unsigned num, Args... args)
// {
//     unsigned i;
//     nElem = 1;
//     isDimInitilized = false;
//     isInitilized = true;
//     nDim = 0;
//     dim_node *ptr_new = new dim_node;
//     head = ptr = ptr_new;
//     head->value = num;
//     head->next = NULL;
//     nDim++;
//     addDimensions(args...);
//     i = nDim - 1;
//     if (!isDimInitilized)
//     {
//         isDimInitilized = true;
//         dimension = new unsigned[nDim];
//         arr_dim = new unsigned[nDim];
//         ptr = head;
//         while (ptr)
//         {
//             dimension[i--] = ptr->value;
//             prev = ptr;
//             ptr = ptr->next;
//             delete prev;
//         }
//     }
//     for (i = 0; i < nDim; i++)
//         nElem *= dimension[i];
//     data = new T[nElem];
//     std::memset(data, 0, nElem * sizeof(T));
// }

template <typename T>
ndarray<T>::ndarray(unsigned n, const unsigned *arr, DataType d_type) {

  this->nDim = n;
  this->nElem = 1;
  this->dimension = new unsigned[this->nDim];
  this->arr_dim = new unsigned[this->nDim];
  this->isDimInitilized = true;
  this->tensor_type = d_type;
#undef data_type

  for (int i = 0; i < this->nDim; i++) {
    this->dimension[i] = arr[i];
    nElem *= dimension[i];
  }
  data_ptr = new EnumToDataType<tf_bfloat16>::type[nElem];

  data = new T[nElem];
  std::memset(data, 0, nElem * sizeof(T));
}

template <typename T> ndarray<T>::~ndarray() {
  // printDimensions();
  // std::cout << "Destroyer " << isInitilized << ", " << isDimInitilized <<
  // "\n"; std::cout << "data: " << data << " dim ptr: " << dimension << " arr
  // ptr: " << arr_dim << "\n";

  // std::cout << "destroyer destroying data for object:" << obj_name << " " <<
  // data << " " << dimension << " " << arr_dim <<  "!\n";
  if (data) {
    delete[] data;
    data = nullptr;
    // std::cout << "freed data memory \n ";
  }

  if (dimension) {
    delete[] dimension;
    dimension = nullptr;
    // std::cout << "freed dimension memory \n ";
  }

  if (arr_dim) {
    delete[] arr_dim;
    arr_dim = nullptr;
    // std::cout << "freed arr_dim memory \n ";
  }
  // std::cout << "destroyed!\n";
}

template <typename T> void ndarray<T>::addDimensions(unsigned w) {
  dim_node *ptr_new = new dim_node;

  ptr->next = ptr_new;
  ptr_new->value = w;
  ptr_new->next = NULL;
  ptr = ptr->next;
  nDim++;
}

template <typename T> void ndarray<T>::addDimensions(const unsigned *w) {
  isDimInitilized = true;
  nDim = head->value;
  dimension = new unsigned[nDim];
  arr_dim = new unsigned[nDim];
  for (unsigned i = 0; i < nDim; i++) {
    dimension[i] = w[i];
  }

  delete head;
}

// template <typename T>
// template <typename... args>
// void ndarray<T>::addDimensions(unsigned w, args... Args) {
//   addDimensions(w);
//   addDimensions(Args...);
// }

template <typename T> const unsigned *ndarray<T>::getDimensions() const {
  return dimension;
}

template <typename T> unsigned ndarray<T>::getNoOfDimensions() const {
  return nDim;
}

template <typename T> const unsigned ndarray<T>::getNoOfElem() const {
  return nElem;
}

template <typename T> T *ndarray<T>::getData() const { return data; }

template <typename T> void ndarray<T>::printDimensions() const {
  std::cout << "[ ";
  for (int i = nDim - 1; i >= 0; i--)
    std::cout << dimension[i] << ", ";
  std::cout << "]";
}

template <typename T> void ndarray<T>::printData() {
  int Elem;

  int *dim;
  dim = new int[nDim];
  for (int i = 0; i < nDim; i++)
    dim[i] = dimension[i];

  for (int i = 0; i < nElem; i++) {
    if (dim[0] == 1)
      std::cout << "[";

    Elem = 1;
    for (int j = 0; j < nDim; j++) {
      Elem *= dim[j];
      if ((i + 1) % Elem == 1)
        std::cout << "[";
    }

    std::cout << "\t";

    // if (type)
    // {
    // printCUDAElement(data + 1);
    // gpu::print<<<1, 1>>>(data + i);
    // cudaDeviceSynchronize();
    // }
    // else
    // {
    std::cout.precision(6);
    std::cout.setf(std::ios::showpoint);
    std::cout << data[i];
    // }

    if ((i + 1) % dim[0] != 0)
      std::cout << ",";

    Elem = 1;
    for (int j = 0; j < nDim; j++) {
      Elem *= dim[j];
      if ((i + 1) % Elem == 0) {
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

template <typename T> void ndarray<T>::printLinearData() {
  for (int i = 0; i < nElem; i++) {
    std::cout << data[i] << ", ";
  }
  std::cout << std::endl;
}

template <typename T> void ndarray<T>::initData(T data) {
  // if (type)
  // {
  T *item = new T[nElem];
  for (int i = 0; i < nElem; i++)
    item[i] = data;
  // GPU_aux<T>::cudaMemoryCopyHostToDevice(this->data,item,nElem);
  // cudaMemcpy(this->data, item, sizeof(T) * nElem, cudaMemcpyHostToDevice);
  delete[] item;
  // }
  // else
  for (int i = 0; i < nElem; i++)
    this->data[i] = data;
}

template <typename T> void ndarray<T>::initData(T *data) {
  // if (type)
  // {
  // GPU_aux<T>::cudaMemmoryCopyToDevice(this->data, data, nElem);
  // }
  // else
  std::memcpy(this->data, data, nElem * sizeof(T));
};

template <typename T> void ndarray<T>::initData(ndarray<T> incData) {

  nDim = incData.getNoOfDimensions();
  nElem = 1;
  dimension = new unsigned[nDim];
  for (unsigned i = 0; i < nDim; i++) {
    dimension[i] = incData.getDimensions()[i];
    nElem *= dimension[i];
  }

  // if (type)
  // {
  // GPU_aux<T>::cudaMemoryCopyHostToDevice(this->data, incData.getData(),
  // nElem); cudaMemcpy(this->data, incData.getData(), sizeof(T) * nElem,
  // cudaMemcpyHostToDevice);
  // }
  // else
  // {
  data = new T[nElem];
  T *ptr = incData.getData();

  for (int i = 0; i < nElem; i++)
    this->data[i] = ptr[i];
  // }
}

template <typename T>
void ndarray<T>::initPartialData(unsigned index, unsigned n, T *data_source) {
  int j = 0;
  // if (type)
  // {
  // GPU_aux<T>::cudaMemoryCopyHostToDevice(data+index, data_source,n);
  // cudaMemcpy((data + index), data_source, sizeof(T) * n,
  // cudaMemcpyHostToDevice);
  // }
  // else
  for (int i = index; i < (index + n); i++)
    data[i] = data_source[j++];
}

template <typename T>
void ndarray<T>::initRandData(double lower_limit, double upper_limit) {
  std::default_random_engine generator;
  std::uniform_real_distribution<T> distribution((0 + lower_limit),
                                                 (1 * upper_limit));

  for (int i = 0; i < nElem; i++) {
    data[i] = distribution(generator);
  }
}

template <typename T> void ndarray<T>::initPreinitilizedData(T *Data) {
  this->data = Data;
}

template <typename T> void ndarray<T>::copyData(T *data) {
  for (int i = 0; i < nElem; i++)
    data[i] = this->data[i];
  std::cout << std::endl;
}

template <typename T>
void ndarray<T>::resetDimensions(unsigned n, unsigned *arr) {

  nDim = n;
  nElem = 1;
  dimension = new unsigned[nDim];
  for (unsigned i = 0; i < nDim; i++) {
    dimension[i] = arr[i];
    nElem *= dimension[i];
  }

  data = new T[nElem];
}

template <typename T> DataType ndarray<T>::getType() { return tensor_type; }

template <typename T> void ndarray<T>::destroy() {
  if (dimension) {
    delete[] dimension;
    dimension = nullptr;
    // std::cout << "freed dimension memory in destroy\n ";
  }
  if (arr_dim) {
    delete[] arr_dim;
    arr_dim = nullptr;
    // std::cout << "freed arr_dim memory in destroy\n ";
  }
  // std::cout << "destroyed!\n";
}

template <typename T> T ndarray<T>::operator()(unsigned n) {
  unsigned loc = n;
  if (dim_iterator == nDim - 1) {
    for (int i = 0; i < dim_iterator; i++) {
      loc += arr_dim[i] * dimension[i];
    }
    dim_iterator = 0;

    return data[loc];
  } else {
    std::cout << "Invalid indexing.\n";
    return (T)0;
  }
}

template <typename T> ndarray<T> &ndarray<T>::operator[](unsigned n) {
  arr_dim[dim_iterator++] = n;
  return *this;
}

template <typename T> void ndarray<T>::setObjName(std::string str) {
  obj_name = str;
}

template <typename T> void ndarray<T>::printNoOfElements() {
  std::cout << nElem << "\n";
}

template <typename T> void ndarray<T>::reshape(unsigned n, unsigned *arr) {
  this->nDim = n;

  delete [] arr;
  delete [] arr_dim;

  this->dimension = new unsigned[this->nDim];
  this->arr_dim = new unsigned[this->nDim];
  this->nElem = 1;

  for (int i = 0; i < this->nDim; i++) {
    this->dimension[i] = arr[i];
    nElem *= dimension[i];
  }
  printDimensions();
}

template <typename T>
ndarray<T> &ndarray<T>::operator=(const ndarray<T> &ndarray) {
  // std::cout << "inside assignment operator" << this << " " << &ndarray <<
  // "\n";
  if (this == &ndarray)
    return *this;
  if (isInitilized)
    destroy();
  isInitilized = true;
  isDimInitilized = true;

  this->nDim = ndarray.getNoOfDimensions();
  dimension = new unsigned[nDim];
  arr_dim = new unsigned[nDim];

  std::memcpy(dimension, ndarray.getDimensions(), nDim * sizeof(unsigned));
  std::memset(arr_dim, 0, nDim * sizeof(unsigned));
  nElem = ndarray.getNoOfElem();

  data = new T[nElem];
  if (ndarray.isInitilized)
    this->initData(ndarray.getData());

  // std::cout << "assignment operator data: " << data << " dim ptr: " <<
  // dimension << " arr ptr: " << arr_dim << "\n";
  return *this;
}

// template class ndarray<std::bfloat16_t>;
// template class ndarray<std::float16_t>;
// template class ndarray<std::float32_t>;
template class ndarray<std::float64_t>;
// template class ndarray<int8_t>;
// template class ndarray<int16_t>;
// template class ndarray<int32_t>;
// template class ndarray<int64_t>;
// template class ndarray<uint8_t>;
// template class ndarray<uint16_t>;
// template class ndarray<uint32_t>;
// template class ndarray<uint64_t>;
