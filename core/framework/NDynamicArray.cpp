#include <absl/log/log.h>
#include <framework/NDynamicArray.h>

template <typename T> ndarray<T>::ndarray() {
  dim_iterator = 0;
  nDim = 0;
  // no_of_gpu = 0;
  dimension = nullptr;
  arr_dim = nullptr;
  data = nullptr;
}

template <typename T> ndarray<T>::ndarray(const ndarray<T> &ndarray) {
  if (isinitialized)
    destroy();
  isinitialized = true;
  isDiminitialized = true;
  this->nDim = ndarray.getNoOfDimensions();

  dimension = new unsigned[nDim];
  arr_dim = new unsigned[nDim];

  std::memcpy(dimension, ndarray.getDimensions(), nDim * sizeof(unsigned));
  std::memset(arr_dim, 0, nDim * sizeof(unsigned));

  nElem = ndarray.getNoOfElem();
  data = new T[nElem];
  if (ndarray.isinitialized)
    this->initData(ndarray.getData());
}

template <typename T>
ndarray<T>::ndarray(unsigned n, const unsigned *arr, DataType d_type) {

  this->nDim = n;
  this->nElem = 1;
  this->dimension = new unsigned[this->nDim];
  this->arr_dim = new unsigned[this->nDim];
  this->isDiminitialized = true;
  this->tensor_type = d_type;

  for (int i = 0; i < this->nDim; i++) {
    this->dimension[i] = arr[i];
    nElem *= dimension[i];
  }

  data = new T[nElem];
  this->isinitialized = true;
  std::memset(data, 0, nElem * sizeof(T));
}

template <typename T>
ndarray<T>::ndarray(unsigned n, const unsigned *arr, DataType d_type,
                    bool is_grad_required) {

  this->nDim = n;
  this->nElem = 1;
  this->dimension = new unsigned[this->nDim];
  this->arr_dim = new unsigned[this->nDim];
  this->isDiminitialized = true;
  this->tensor_type = d_type;

  for (int i = 0; i < this->nDim; i++) {
    this->dimension[i] = arr[i];
    nElem *= dimension[i];
  }

  data = new T[nElem];
  this->isinitialized = true;
  std::memset(data, 0, nElem * sizeof(T));

  this->is_grad_required = is_grad_required;
}

template <typename T>
ndarray<T> &ndarray<T>::operator=(const ndarray<T> &ndarray) {
  if (this == &ndarray)
    return *this;
  if (this->isinitialized)
    destroy();
  isinitialized = true;
  isDiminitialized = true;

  this->nDim = ndarray.nDim;
  dimension = new unsigned[nDim];
  arr_dim = new unsigned[nDim];

  std::memcpy(dimension, ndarray.getDimensions(), nDim * sizeof(unsigned));
  std::memset(arr_dim, 0, nDim * sizeof(unsigned));
  nElem = ndarray.nElem;

  data = new T[nElem];
  if (ndarray.isinitialized)
    this->initData(ndarray.getData());
  return *this;
}

template <typename T> ndarray<T>::ndarray(ndarray<T> &&ndarray) noexcept {
  nDim = ndarray.nDim;
  nElem = ndarray.nElem;
  dimension = ndarray.dimension;
  arr_dim = ndarray.arr_dim;
  dim_iterator = ndarray.dim_iterator;
  isDiminitialized = ndarray.isDiminitialized;
  isinitialized = ndarray.isinitialized;
  data = ndarray.data;
  ndarray.dimension = nullptr;
  ndarray.arr_dim = nullptr;
  ndarray.data = nullptr;
}

template <typename T>
ndarray<T> &ndarray<T>::operator=(ndarray<T> &&ndarray) noexcept {
  if (this != &ndarray) {
    if (isinitialized)
      destroy();
    nDim = ndarray.nDim;
    nElem = ndarray.nElem;
    dimension = ndarray.dimension;
    arr_dim = ndarray.arr_dim;
    dim_iterator = ndarray.dim_iterator;
    isDiminitialized = ndarray.isDiminitialized;
    isinitialized = ndarray.isinitialized;
    data = ndarray.data;
    ndarray.dimension = nullptr;
    ndarray.arr_dim = nullptr;
    ndarray.data = nullptr;
  }
  return *this;
}

template <typename T> ndarray<T>::~ndarray() {
  // std::cout << "Destructor called for object: " << "\n";
  LOG(INFO) << "Destructor called for object: " << "\n";
  destroy();
}

template <typename T> T *ndarray<T>::getData() const { return data; }

template <typename T> DataType ndarray<T>::getDataType() { return tensor_type; }

template <typename T> const unsigned *ndarray<T>::getDimensions() const {
  return dimension;
}

template <typename T> unsigned ndarray<T>::getNoOfDimensions() const {
  return nDim;
}

template <typename T> const unsigned ndarray<T>::getNoOfElem() const {
  return nElem;
}

template <typename T> void ndarray<T>::printDimensions() const {
  std::cout << "[ ";
  for (int i = nDim - 1; i >= 0; i--)
    std::cout << dimension[i] << ", ";
  std::cout << "]";
}

template <typename T> void ndarray<T>::printData() {
  int Elem;

  int *dim;
  dim = new int[this->nDim];
  for (int i = 0; i < this->nDim; i++)
    dim[i] = this->dimension[i];

  for (int i = 0; i < this->nElem; i++) {
    if (dim[0] == 1)
      std::cout << "[";

    Elem = 1;
    for (int j = 0; j < this->nDim; j++) {
      Elem *= dim[j];
      if ((i + 1) % Elem == 1)
        std::cout << "[";
    }

    std::cout << "\t";
    std::cout.precision(6);
    std::cout.setf(std::ios::showpoint);
    std::cout << data[i];
    // }

    if ((i + 1) % dim[0] != 0)
      std::cout << ",";

    Elem = 1;
    for (int j = 0; j < this->nDim; j++) {
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
  delete[] dim;
}

template <typename T> void ndarray<T>::printLinearData() {
  for (int i = 0; i < nElem; i++) {
    std::cout << data[i] << ", ";
  }
  std::cout << std::endl;
}

template <typename T> void ndarray<T>::initData(T data) {
  T *item = new T[nElem];
  for (int i = 0; i < nElem; i++)
    item[i] = data;
  delete[] item;
  for (int i = 0; i < nElem; i++)
    this->data[i] = data;
}

template <typename T> void ndarray<T>::initData(T *data) {
  if (data) {
    std::memcpy(this->data, data, nElem * sizeof(T));
  } else if (data == nullptr) {
    std::cout << "Data pointer is null, cannot initialize data.\n";
    return;
  }
  if (nElem == 0) {
    std::cout << "No elements to initialize.\n";
    return;
  }
};

template <typename T> void ndarray<T>::initData(ndarray<T> incData) {

  nDim = incData.getNoOfDimensions();
  nElem = 1;
  dimension = new unsigned[nDim];
  for (unsigned i = 0; i < nDim; i++) {
    dimension[i] = incData.getDimensions()[i];
    nElem *= dimension[i];
  }

  data = new T[nElem];
  T *ptr = incData.getData();

  for (int i = 0; i < nElem; i++)
    this->data[i] = ptr[i];
}

template <typename T>
void ndarray<T>::initPartialData(unsigned index, unsigned n, T *data_source) {
  int j = 0;
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

template <typename T> void ndarray<T>::initPreinitializedData(T *Data) {
  this->data = Data;
}

template <typename T> bool ndarray<T>::isGradRequired() {
  return this->is_grad_required;
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
  if (data) {
    delete[] data;
    data = nullptr;
  }
  if (dimension) {
    delete[] dimension;
    dimension = nullptr;
  }
  if (arr_dim) {
    delete[] arr_dim;
    arr_dim = nullptr;
  }
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

template <typename T>
void ndarray<T>::reshape(unsigned n, const unsigned *arr) {
  if (this->nDim < n) {
    this->nDim = n;
    delete[] this->dimension;
    delete[] this->arr_dim;
    this->dimension = new unsigned[this->nDim];
    this->arr_dim = new unsigned[this->nDim];
  } else {
    this->nDim = n;
  }

  unsigned temp_nElem = 1;
  for (int i = 0; i < this->nDim; i++) {
    this->dimension[i] = arr[i];
    temp_nElem *= dimension[i];
  }

  if (this->nElem < temp_nElem) {
    delete[] data;
    this->nElem = temp_nElem;
    data = new T[this->nElem];
  } else {
    this->nElem = temp_nElem;
  }
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
