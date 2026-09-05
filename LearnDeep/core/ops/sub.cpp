#ifdef ENABLE_CUDA
#include <core/LAS/gpu_interface.cuh>
#endif

// Library Headers
#include "kernelmanager.h"
#include "opskernel.h"
#include <core/LAS/CPULibrary.h>
#include <core/LAS/avx2_micro_kernels.h>
#include <core/framework/MathLibrary.h>

void Opssub::addGradGraph(Graph *gradient_graph) {
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

  if (!this->isBroadCast) {

    // for first input we can directly push the incoming gradient to previous
    // layer
    this->outgoing_gradients.push_back(this->incoming_gradient);

    // for second input we need to calculate broadcast dimentions and then do a
    // reduction sum then push the incoming gradient to previous layer.

    Tensor<std::float64_t> *tensor_negative_ones =
        new Tensor<std::float64_t>(*this->inputs[1]);
    tensor_negative_ones->initData(-1.0);

    Tensor<std::float64_t> *negative_tensor_output =
        new Tensor<std::float64_t>(*this->inputs[1]);

    tensor_ptr[0] = this->incoming_gradient;
    tensor_ptr[1] = tensor_negative_ones;

    Ops *ops_mul = new Opsmul();
    ops_mul->initializeinputs(tensor_ptr);
    ops_mul->initializeoutput(negative_tensor_output);

    gradient_graph->addGradientNode(ops_mul);
    gradient_graph->addGradientNode(this->incoming_gradient);
    gradient_graph->addGradientNode(tensor_negative_ones);
    gradient_graph->addGradientNode(negative_tensor_output);
    gradient_graph->addGradientEdge(this->incoming_gradient, ops_mul);
    gradient_graph->addGradientEdge(tensor_negative_ones, ops_mul);
    gradient_graph->addGradientEdge(ops_mul, negative_tensor_output);

    this->outgoing_gradients.push_back(negative_tensor_output);

  } else {
    // for first input we can directly push the incoming gradient to previous
    // layer
    this->outgoing_gradients.push_back(this->incoming_gradient);

    // for second input we need to calculate broadcast dimentions and then do a
    // reduction sum then push the incoming gradient to previous layer.
    Tensor<std::float64_t> *reduction_result = new Tensor<std::float64_t>();
    Ops *opsreducesum = new Opsreducesum();
    opsreducesum->initializeinputs(&this->incoming_gradient);
    opsreducesum->initializeReductionDims(this->broadCastAxies.size(),
                                          this->broadCastAxies.data());
    opsreducesum->initializeoutput(reduction_result);

    Tensor<std::float64_t> *tensor_negative_ones =
        new Tensor<std::float64_t>(*opsreducesum->getoutput());
    tensor_negative_ones->initData(-1.0);
    Tensor<std::float64_t> *tensor_multiple_result =
        new Tensor<std::float64_t>();

    Ops *opsmul = new Opsmul();

    tensor_ptr[0] = reduction_result;
    tensor_ptr[1] = tensor_negative_ones;
    opsmul->initializeinputs(tensor_ptr);
    opsmul->initializeoutput(tensor_multiple_result);

    gradient_graph->addGradientNode(opsreducesum);
    gradient_graph->addGradientNode(this->incoming_gradient);
    gradient_graph->addGradientNode(reduction_result);
    gradient_graph->addGradientEdge(this->incoming_gradient, opsreducesum);
    gradient_graph->addGradientEdge(opsreducesum, reduction_result);

    gradient_graph->addGradientNode(opsmul);
    gradient_graph->addGradientNode(tensor_negative_ones);
    gradient_graph->addGradientNode(tensor_multiple_result);
    gradient_graph->addGradientEdge(reduction_result, opsmul);
    gradient_graph->addGradientEdge(tensor_negative_ones, opsmul);
    gradient_graph->addGradientEdge(opsmul, tensor_multiple_result);

    this->outgoing_gradients.push_back(tensor_multiple_result);
  }
}

void Opssub::compute() {
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
              "b. and also not broadcasting compatable");
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

void Opssub::initializeinputs(Tensor<std::float64_t> **inputs) {
  this->inputs.push_back(inputs[0]);
  this->inputs.push_back(inputs[1]);
}

void Opssub::initializeoutput(Tensor<std::float64_t> *output) {
  this->output = output;

  *(this->output) = *(inputs[0]);
}

void Opssub::printinputs() {
  for (unsigned i = 0; i < 2; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opssub::printoutput() {
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}

Tensor<std::float64_t> *
Opssub::getOutgoingGradientTensor(Tensor<std::float64_t> *gradient_input) {
  auto it = std::find(inputs.begin(), inputs.end(), gradient_input);
  Tensor<std::float64_t> *ptr = nullptr;
  if (inputs.end() != it) {
    int idx = std::distance(inputs.begin(), it);
    ptr = outgoing_gradients[idx];
  }
  return ptr;
}

Tensor<std::float64_t> *
Opssub::getIncomingGradientTensor(Tensor<std::float64_t> *tensor) {
  return incoming_gradient;
}

void Opssub::kernel_dispatch(std::float64_t **ptr, const unsigned nDimA,
                             const unsigned *dimA, const unsigned nDimB,
                             const unsigned *dimB, const bool isBroadCast) {
  KernelType kernel = get_global_kernel();
#ifdef ENABLE_CUDA
  switch (kernel) {
  case KernelType::GPU: {
    double *d_arr[3];
    d_arr[0] = reinterpret_cast<double *>(ptr[0]);
    d_arr[1] = reinterpret_cast<double *>(ptr[1]);
    d_arr[2] = reinterpret_cast<double *>(ptr[2]);
    gpu::gpu_mat_sub_broadcast_f64(d_arr, nDimA, dimA, nDimB, dimB,
                                   isBroadCast);
    break;
  }
  case KernelType::AVX2:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_sub_broadcast_f64(ptr, nDimA, dimA, nDimB, dimB, isBroadCast);
    } else {
      throw std::runtime_error("AVX2 not supported on this CPU");
    }
    break;
  case KernelType::CPU_SCALAR:
  case KernelType::AUTO:
  default: {
    cpu::__msub_broadcast(ptr, nDimA, dimA, nDimB, dimB, isBroadCast);
    break;
  }
  }
#else
  switch (kernel) {
  case KernelType::GPU:
    throw std::runtime_error("GPU kernel requested but CUDA not enabled");
  case KernelType::AVX2:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_sub_broadcast_f64(ptr, nDimA, dimA, nDimB, dimB, isBroadCast);
    } else {
      throw std::runtime_error("AVX2 not supported on this CPU");
    }
    break;
  case KernelType::CPU_SCALAR:
    cpu::__msub_broadcast(ptr, nDimA, dimA, nDimB, dimB, isBroadCast);
    break;
  case KernelType::AUTO:
  default:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_sub_broadcast_f64(ptr, nDimA, dimA, nDimB, dimB, isBroadCast);
    } else {
      cpu::__msub_broadcast(ptr, nDimA, dimA, nDimB, dimB, isBroadCast);
    }
    break;
  }
#endif
}
