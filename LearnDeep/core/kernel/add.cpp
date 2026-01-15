#include <cstddef>
#include <elf.h>
#ifdef CUDA_ENABLED
#include <core/LAS/gpu_interface.cuh>
#endif

// Library Headers
#include "opskernel.h"
#include <core/LAS/CPULibrary.h>
#include <core/LAS/avx2_micro_kernels.h>
#include <core/framework/MathLibrary.h>

void Opsadd::addGradGraph(Graph *gradient_graph) {
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

  for (unsigned i = 0; i < 2; i++)
    this->outgoing_gradients.push_back(this->incoming_gradient);
}

void Opsadd::compute() {
  unsigned j = 0;
  if (!this->isPreInitializationDone) {
    for (unsigned i = 0; i < inputs[0]->getNoOfDimensions(); ++i) {
      if (j < inputs[1]->getNoOfDimensions()) {
        if (inputs[0]->getDimensions()[i] == inputs[1]->getDimensions()[j])
          broadCastDimensionSizes.push_back(inputs[1]->getDimensions()[j++]);
        else if (inputs[0]->getDimensions()[i] !=
                     inputs[1]->getDimensions()[j] &&
                 inputs[1]->getDimensions()[j] == 1) {
          broadCastDimensionSizes.push_back(inputs[1]->getDimensions()[j++]);
          this->isBroadCast = true;
        } else {
          throw std::runtime_error(
              "Dimension of input a is not a match with input "
              "b. and also not broad casting compatable");
        }
      } else {
        broadCastDimensionSizes.push_back(1);
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

void Opsadd::initializeinputs(Tensor<std::float64_t> **inputs) {
  this->inputs.push_back(inputs[0]);
  this->inputs.push_back(inputs[1]);
}

void Opsadd::initializeoutput(Tensor<std::float64_t> *output) {
  this->output = output;
  bool flag = false;
  if (this->inputs[0]->getNoOfDimensions() ==
      this->output->getNoOfDimensions()) {
    for (int i = 0; i < this->inputs[0]->getNoOfDimensions(); i++) {
      if (this->output->getDimensions()[i] !=
          this->inputs[0]->getDimensions()[i]) {
        flag = true;
        break;
      }
    }
  } else {
    flag = true;
  }

  if (flag)
    this->output->reshape(inputs[0]->getNoOfDimensions(),
                          inputs[0]->getDimensions());

  // *(this->output) = *(inputs[0]);
}

void Opsadd::printinputs() {
  unsigned i;
  for (i = 0; i < 2; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opsadd::printoutput() {
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}

Tensor<std::float64_t> *
Opsadd::getOutgoingGradientTensor(Tensor<std::float64_t> *gradient_input) {
  auto it = std::find(inputs.begin(), inputs.end(), gradient_input);
  Tensor<std::float64_t> *ptr = nullptr;
  if (inputs.end() != it) {
    int idx = std::distance(inputs.begin(), it);
    ptr = outgoing_gradients[idx];
  }
  return ptr;
}

Tensor<std::float64_t> *
Opsadd::getIncomingGradientTensor(Tensor<std::float64_t> *tensor) {
  return incoming_gradient;
}

void Opsadd::kernel_dispatch(std::float64_t **ptr, const unsigned nDimA,
                             const unsigned *dimA, const unsigned nDimB,
                             const unsigned *dimB, const bool isBroadCast) {

#ifdef CUDA_ENABLED

  double *d_arr[3];
  d_arr[0] = reinterpret_cast<double *>(ptr[0]);
  d_arr[1] = reinterpret_cast<double *>(ptr[1]);
  d_arr[2] = reinterpret_cast<double *>(ptr[2]);
  gpu::gpu_mat_add_broadcast_f64(d_arr, nDimA, dimA, nDimB, dimB, isBroadCast);

#else

  if (__builtin_cpu_supports("avx2")) {
    avx2::avx2_add_broadcast_f64(ptr, nDimA, dimA, nDimB, dimB, isBroadCast);
  } else {
    cpu::__madd_broadcast(ptr, nDimA, dimA, nDimB, dimB, isBroadCast);
  }

#endif
}
