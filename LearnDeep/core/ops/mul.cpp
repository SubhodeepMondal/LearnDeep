#include <iterator>
#ifdef ENABLE_CUDA
#include <core/LAS/gpu_interface.cuh>
#endif

// Library Headers
#include "kernelmanager.h"
#include <core/LAS/CPULibrary.h>
#include <core/LAS/avx2_micro_kernels.h>
#include <core/framework/MathLibrary.h>
#include <core/ops/opskernel.h>

void Opsmul::compute() {

  unsigned j = 0;
  if (!this->isPreInitializationDone) {
    for (unsigned i = 0; i < inputs[0]->getNoOfDimensions(); ++i) {
      if (j < inputs[1]->getNoOfDimensions()) {
        if (inputs[0]->getDimensions()[i] == inputs[1]->getDimensions()[j]) {
          this->broadCastDimensionSizes.push_back(
              inputs[1]->getDimensions()[j++]);
        } else if (inputs[0]->getDimensions()[i] !=
                       inputs[1]->getDimensions()[j] &&
                   inputs[1]->getDimensions()[j] == 1) {
          broadCastDimensionSizes.push_back(inputs[1]->getDimensions()[j++]);
          this->broadCastAxies.push_back(i);
          this->isBroadCast = true;
        } else {
          throw std::runtime_error(
              "Dimension of input a is not a match with input "
              "b. and also not broad casting compatable");
        }
      } else {
        this->broadCastAxies.push_back(i);
        this->broadCastDimensionSizes.push_back(1);
        this->isBroadCast = true;
      }
    }
    this->isPreInitializationDone = true;
  }

  std::float64_t *ptr[3];
  ptr[0] = inputs[0]->getData();
  ptr[1] = inputs[1]->getData();
  ptr[2] = output->getData();

  /* kernel dispatch*/
  this->kernel_dispatch(ptr, inputs[0]->getNoOfDimensions(),
                        inputs[0]->getDimensions(),
                        broadCastDimensionSizes.size(),
                        broadCastDimensionSizes.data(), isBroadCast);
}

void Opsmul::addGradGraph(Graph *gradient_graph) {
  // .......... reverse mode autodiff graph .........
  //
  //             [inputs[n]]
  //                 |
  //        [temp_grad_tensor[n]]  *  [[incoming_gradients]...]
  //                           [[add]...]
  //                              |
  //                      [output_gradient]
  //
  // ........................ End .....................

  Tensor<std::float64_t> *tensor_ptr[2];
  std::vector<Tensor<std::float64_t> *> incoming_gradients =
      gradient_graph->getGradient(this);

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

  Tensor<std::float64_t> *temp_grad_tensors;
  for (unsigned i = 0; i < 2; i++) {

    // Finding d/dx[i] for multiplication operation
    //  f(x[i]) = x[i] * x_b
    //  f'(x[i]) = x_b
    temp_grad_tensors =
        new Tensor<std::float64_t>(*this->inputs[(2 - i - 1) % 2]);
    // end of Finding d/dx[i]

    // graph setup for d/dx[i] * z'
    Ops *ops_mul = new Opsmul;
    tensor_ptr[0] = temp_grad_tensors;
    tensor_ptr[1] = this->incoming_gradient;

    // input initialization
    ops_mul->initializeinputs(tensor_ptr);
    gradient_graph->addGradientNode(ops_mul);
    gradient_graph->addGradientNode(tensor_ptr[0]);
    gradient_graph->addGradientNode(tensor_ptr[1]);
    gradient_graph->addGradientEdge(tensor_ptr[0], ops_mul);
    gradient_graph->addGradientEdge(tensor_ptr[1], ops_mul);

    // output initialization
    this->outgoing_gradients[i] = new Tensor<std::float64_t>(*this->inputs[i]);
    ops_mul->initializeoutput(this->outgoing_gradients[i]);
    gradient_graph->addGradientNode(this->outgoing_gradients[i]);
    gradient_graph->addGradientEdge(ops_mul, this->outgoing_gradients[i]);
    // End of d/dx[i] * z'
  }
}

void Opsmul::initializeinputs(Tensor<std::float64_t> **inputs) {
  unsigned i;
  // this->inputs = new Tensor<std::float64_t> *[this->no_of_inputs];

  for (i = 0; i < 2; i++) {
    this->inputs.push_back(inputs[i]);
  }
}

void Opsmul::initializeoutput(Tensor<std::float64_t> *output) {
  this->output = output;
  *(this->output) = *(inputs[0]);
}

void Opsmul::printinputs() {
  unsigned i;
  for (i = 0; i < 2; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opsmul::printoutput() {
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}

Tensor<std::float64_t> *
Opsmul::getOutgoingGradientTensor(Tensor<std::float64_t> *gradient_input) {
  int i, it;
  bool flag = false;
  for (i = 0; i < 2; i++)
    if (this->inputs[i] == gradient_input) {
      it = i;
      flag = true;
      break;
    }

  if (flag) {
    // LOG(INFO) << "Requested gradint for the tensor found.\n";
    return this->outgoing_gradients[it];
  } else {
    // LOG(FATAL) << "Requested gradint for the tensor doesn't exist.\n";
    return NULL;
  }
}

void Opsmul::kernel_dispatch(std::float64_t **ptr, const unsigned nDimA,
                             const unsigned *dimA, const unsigned nDimB,
                             const unsigned *dimB, const bool isBroadCast) {

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
    double *d_arr[3];
    d_arr[0] = reinterpret_cast<double *>(ptr[0]);
    d_arr[1] = reinterpret_cast<double *>(ptr[1]);
    d_arr[2] = reinterpret_cast<double *>(ptr[2]);
    gpu::gpu_mat_hadamard_mul_broadcast_f64(d_arr, nDimA, dimA, nDimB, dimB,
                                            isBroadCast);
  }
#else
    throw std::runtime_error("GPU kernel requested but CUDA not enabled");
#endif
  break;

  case KernelType::AVX2:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_mul_broadcast_f64(ptr, nDimA, dimA, nDimB, dimB, isBroadCast);
    } else {
      throw std::runtime_error("AVX2 not supported on this CPU");
    }
    break;

  case KernelType::CPU_SCALAR:
    cpu::__mmul_broadcast(ptr, nDimA, dimA, nDimB, dimB, isBroadCast);
    break;

  case KernelType::AUTO:
  default:
#ifdef ENABLE_CUDA
  {
    double *d_arr[3];
    d_arr[0] = reinterpret_cast<double *>(ptr[0]);
    d_arr[1] = reinterpret_cast<double *>(ptr[1]);
    d_arr[2] = reinterpret_cast<double *>(ptr[2]);
    gpu::gpu_mat_hadamard_mul_broadcast_f64(d_arr, nDimA, dimA, nDimB, dimB,
                                            isBroadCast);
  }
#else
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_mul_broadcast_f64(ptr, nDimA, dimA, nDimB, dimB, isBroadCast);
    } else {
      cpu::__mmul_broadcast(ptr, nDimA, dimA, nDimB, dimB, isBroadCast);
    }
#endif
  break;
  }
}