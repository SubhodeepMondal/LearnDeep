#include "opskernel.h"
#ifdef ENABLE_CUDA
#include <core/LAS/gpu_interface.cuh>
#endif

// Thirdparty Library
#include <absl/log/log.h>

// Library Headers
#include "kernelmanager.h"
#include <core/LAS/CPULibrary.h>
#include <core/LAS/avx2_micro_kernels.h>
#include <core/framework/MathLibrary.h>
#include <core/ops/opskernel.h>

// standard Libery
#include <algorithm>

Opsreducesum::Opsreducesum(bool keep_dims) : keep_dims(keep_dims) {}

Opsreducesum::~Opsreducesum() {
  if (this->arr)
    delete[] arr;
  if (this->temp_input)
    delete[] this->temp_input;
  if (this->temp_output)
    delete[] this->temp_output;
}

void Opsreducesum::addGradGraph(Graph *gradient_graph) {
  // .......... reverse mode autodiff graph .........
  //
  //             [inputs[n]]
  //                 |
  //             [[add]...]
  //                 |
  //          [output_gradient]
  //
  // ........................ End .....................

  std::vector<Tensor<std::float64_t> *> incoming_gradients =
      gradient_graph->getGradient(this);
  Tensor<std::float64_t> *tensor_ptr[2];

  // graph setup for accumulating incoming gradients y' = sum ( z' )
  if (incoming_gradients.size()) {
    Tensor<std::float64_t> *intermediate_gradient_sum;

    intermediate_gradient_sum = new Tensor<std::float64_t>(*this->output);
    intermediate_gradient_sum->initData(0.0);
    int i = 0;
    for (Tensor<std::float64_t> *inc_grad_tensor : incoming_gradients) {

      // input initialization
      tensor_ptr[0] = intermediate_gradient_sum;
      tensor_ptr[1] = inc_grad_tensor;

      Ops *ops_add = new Opsadd;
      ops_add->initializeinputs(tensor_ptr);

      gradient_graph->addGradientNode(ops_add);
      gradient_graph->addGradientNode(tensor_ptr[0]);
      gradient_graph->addGradientNode(tensor_ptr[1]);
      gradient_graph->addGradientEdge(tensor_ptr[0], ops_add);
      gradient_graph->addGradientEdge(tensor_ptr[1], ops_add);

      // output initialization
      intermediate_gradient_sum = new Tensor<std::float64_t>(*this->output);
      intermediate_gradient_sum->initData(0.0);

      ops_add->initializeoutput(intermediate_gradient_sum);
      gradient_graph->addGradientNode(intermediate_gradient_sum);
      gradient_graph->addGradientEdge(ops_add, intermediate_gradient_sum);
    }
    this->incoming_gradient = intermediate_gradient_sum;
  } else {
    this->incoming_gradient = new Tensor<std::float64_t>(*this->output);
    this->incoming_gradient->initData(1.0);
  }

  // for second input we need to calculate broadcast dimentions and then do a
  // reduction sum then push the incoming gradient to previous layer.
  Tensor<std::float64_t> *broadcasting_parent =
      new Tensor<std::float64_t>(*inputs[0]);
  broadcasting_parent->initData(0.0);

  Ops *opsadd = new Opsadd();
  tensor_ptr[0] = broadcasting_parent;
  tensor_ptr[1] = this->incoming_gradient;
  opsadd->initializeinputs(tensor_ptr);

  gradient_graph->addGradientNode(opsadd);
  gradient_graph->addGradientNode(this->incoming_gradient);
  gradient_graph->addGradientNode(broadcasting_parent);
  gradient_graph->addGradientEdge(this->incoming_gradient, opsadd);
  gradient_graph->addGradientEdge(broadcasting_parent, opsadd);

  Tensor<std::float64_t> *broadcasting_result =
      new Tensor<std::float64_t>(*inputs[0]);

  opsadd->initializeoutput(broadcasting_result);
  gradient_graph->addGradientEdge(opsadd, broadcasting_result);
  this->outgoing_gradients.push_back(broadcasting_result);
}

void Opsreducesum::compute() {

  unsigned total_elem = this->inputs[0]->getNoOfElem();
  if (this->isFirstRun) {
    arr = new unsigned[this->inputs[0]->getNoOfDimensions() + 2];
    temp_input = new std::float64_t[total_elem];
    temp_output = new std::float64_t[total_elem];
    this->isFirstRun = false;
  }

  std::memcpy(temp_output, this->inputs[0]->getData(),
              this->inputs[0]->getNoOfElem() * sizeof(std::float64_t));
  unsigned no_of_temp_dims = this->inputs[0]->getNoOfDimensions();
  std::float64_t *ptr[2];
  ptr[0] = temp_input;
  ptr[1] = temp_output;

  arr[0] = no_of_temp_dims;
  for (unsigned i = 0; i < this->inputs[0]->getNoOfDimensions(); i++) {
    arr[i + 1] = this->inputs[0]->getDimensions()[i];
  }

  for (unsigned i = 0; i < no_of_reduction_dim; i++) {

    std::memcpy(temp_input, temp_output, total_elem * sizeof(std::float64_t));
    unsigned axis = reduction_dims[i] - i;
    arr[no_of_temp_dims + 1] = axis;

    kernel_dispatch(ptr, arr);

    unsigned k = 1;
    for (unsigned j = 1; j <= no_of_temp_dims; j++)
      if (axis != j - 1)
        arr[k++] = arr[j];
    total_elem /= this->inputs[0]->getDimensions()[reduction_dims[i]];

    no_of_temp_dims -= 1;
    arr[0] = no_of_temp_dims;
  }
  this->output->initData(temp_output);
}

void Opsreducesum::initializeinputs(Tensor<std::float64_t> **inputs) {

  this->inputs.push_back(inputs[0]);
}

void Opsreducesum::initializeReductionDims(const unsigned n,
                                           const unsigned *arr) {
  unsigned i;
  this->no_of_reduction_dim = n;

  // reduction_dims = new unsigned[n];
  for (i = 0; i < n; i++)
    this->reduction_dims.push_back(arr[i]);

  std::sort(this->reduction_dims.begin(), this->reduction_dims.end());
}

void Opsreducesum::initializeoutput(Tensor<std::float64_t> *output) {
  std::vector<unsigned> resultent_dims;
  this->output = output;

  unsigned j = 0;
  for (unsigned i = 0; i < inputs[0]->getNoOfDimensions(); i++) {
    if (i != this->reduction_dims[j]) {
      resultent_dims.push_back(inputs[0]->getDimensions()[i]);
    } else if (this->keep_dims) {
      resultent_dims.push_back(1);
      j++;
    } else {
      j++;
    }
  }
  this->output->reshape(resultent_dims.size(), resultent_dims.data());
}

void Opsreducesum::printinputs() {
  unsigned i;
  for (i = 0; i < 1; i++) {
    LOG(INFO) << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opsreducesum::printoutput() {
  LOG(INFO) << "output:\n";
  output->printData();
  LOG(INFO) << "\n";
}

Tensor<std::float64_t> *Opsreducesum::getOutgoingGradientTensor(
    Tensor<std::float64_t> *gradient_input) {
  auto it = std::find(inputs.begin(), inputs.end(), gradient_input);
  Tensor<std::float64_t> *ptr = nullptr;
  if (inputs.end() != it) {
    int idx = std::distance(inputs.begin(), it);
    ptr = outgoing_gradients[idx];
  }
  return ptr;
}

void Opsreducesum::kernel_dispatch(std::float64_t **ptr, unsigned *arr) {
  KernelType kernel = get_global_kernel();
#ifdef ENABLE_CUDA
  bool gpu_available = true;
#else
  bool gpu_available = false;
#endif

  switch (kernel) {

  case KernelType::GPU:
#ifdef ENABLE_CUDA
  {
    double *d_arr[2];
    d_arr[0] = reinterpret_cast<double *>(ptr[0]);
    d_arr[1] = reinterpret_cast<double *>(ptr[1]);
    gpu::gpu_reduce_sum_f64(d_arr, arr);
  }
#else
    throw std::runtime_error("GPU kernel requested but CUDA not enabled");
#endif
  break;

  case KernelType::AVX2:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_reduce_sum_f64(ptr, arr);
    } else {
      throw std::runtime_error("AVX2 not supported on this CPU");
    }
    break;

  case KernelType::CPU_SCALAR:
    cpu::__mreducesum(ptr, arr);
    break;

  case KernelType::AUTO:
  default:
#ifdef ENABLE_CUDA
  {
    double *d_arr[2];
    d_arr[0] = reinterpret_cast<double *>(ptr[0]);
    d_arr[1] = reinterpret_cast<double *>(ptr[1]);
    gpu::gpu_reduce_sum_f64(d_arr, arr);
  }
#else
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_reduce_sum_f64(ptr, arr);
    } else {
      cpu::__mreducesum(ptr, arr);
    }
#endif
  break;
  }
}
