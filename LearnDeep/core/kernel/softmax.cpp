#ifdef ENABLE_CUDA
#include <core/LAS/gpu_interface.cuh>
#endif

// Library Headers
#include "kernelmanager.h"
#include "opskernel.h"
#include <core/LAS/CPULibrary.h>
#include <core/LAS/avx2_micro_kernels.h>
#include <core/framework/MathLibrary.h>

void Opssoftmax::addGradGraph(Graph *gradient_graph) {
  if (!gradient_graph)
    throw std::runtime_error("Opssoftmax::addGradGraph received null graph");
  if (this->inputs.empty() || !this->inputs[0] || !this->output)
    throw std::runtime_error(
        "Opssoftmax::addGradGraph called before inputs/output initialization");

  // .......... reverse mode autodiff graph .........
  //        [softmax]  *  [[incoming_gradients]...]
  //                   |[T1]
  //                 reduce_sum(T1, axis)
  //                   |[T2]
  //                 broadcast_sub(incoming_grads, T2)
  //                   |[T3]
  //                 mul(T3 * softmax);
  // ........................ End .....................

  Tensor<std::float64_t> *tensor_ptr[2];
  std::vector<Tensor<std::float64_t> *> incoming_gradients =
      gradient_graph->getGradient(this);

  // graph setup for accumulating incoming gradients y' = sum ( z' )
  if (incoming_gradients.size()) {

    // graph setup for  x' = sum ( z' * d/dx )
    Tensor<std::float64_t> *intermediate_gradient_sum;
    intermediate_gradient_sum = new Tensor<std::float64_t>(*this->output);
    intermediate_gradient_sum->initData(0.0);

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

  Tensor<std::float64_t> *temp_grad_tensors[2];

  // initializing temp variables for grad calculation
  temp_grad_tensors[0] = this->output;
  temp_grad_tensors[1] = this->incoming_gradient;
  Tensor<std::float64_t> *mul_output =
      new Tensor<std::float64_t>(*this->inputs[0]);

  // multiplication
  Ops *ops_mul = new Opsmul;
  ops_mul->initializeinputs(temp_grad_tensors);
  ops_mul->initializeoutput(mul_output);

  gradient_graph->addGradientNode(temp_grad_tensors[0]);
  gradient_graph->addGradientNode(temp_grad_tensors[1]);
  gradient_graph->addGradientNode(mul_output);
  gradient_graph->addGradientNode(ops_mul);

  gradient_graph->addGradientEdge(temp_grad_tensors[0], ops_mul);
  gradient_graph->addGradientEdge(temp_grad_tensors[1], ops_mul);
  gradient_graph->addGradientEdge(ops_mul, mul_output);

  // reduce sum
  Ops *ops_reduce_sum = new Opsreducesum;
  Tensor<std::float64_t> *reduce_output = new Tensor(*this->inputs[0]);

  ops_reduce_sum->initializeinputs(&mul_output);
  ops_reduce_sum->initializeReductionDims(1, &this->axis);
  ops_reduce_sum->initializeoutput(reduce_output);

  gradient_graph->addGradientNode(mul_output);
  gradient_graph->addGradientNode(reduce_output);
  gradient_graph->addGradientNode(ops_reduce_sum);

  gradient_graph->addGradientEdge(mul_output, ops_reduce_sum);
  gradient_graph->addGradientEdge(ops_reduce_sum, reduce_output);

  // broadcast substraction
  Ops *ops_sub = new Opssub;
  tensor_ptr[0] = this->incoming_gradient;
  tensor_ptr[1] = reduce_output;

  ops_sub->initializeinputs(tensor_ptr);
  gradient_graph->addGradientNode(ops_sub);
  gradient_graph->addGradientNode(tensor_ptr[0]);
  gradient_graph->addGradientNode(tensor_ptr[1]);
  gradient_graph->addGradientEdge(tensor_ptr[0], ops_sub);
  gradient_graph->addGradientEdge(tensor_ptr[1], ops_sub);

  Tensor<std::float64_t> *sub_output =
      new Tensor<std::float64_t>(*this->inputs[0]);
  ops_sub->initializeoutput(sub_output);
  gradient_graph->addGradientNode(sub_output);
  gradient_graph->addGradientEdge(ops_sub, sub_output);

  // multiplication
  Ops *ops_mul_2 = new Opsmul;
  tensor_ptr[0] = this->output;
  tensor_ptr[1] = sub_output;

  // input initialization
  ops_mul_2->initializeinputs(tensor_ptr);
  gradient_graph->addGradientNode(ops_mul_2);
  gradient_graph->addGradientNode(tensor_ptr[0]);
  gradient_graph->addGradientNode(tensor_ptr[1]);
  gradient_graph->addGradientEdge(tensor_ptr[0], ops_mul_2);
  gradient_graph->addGradientEdge(tensor_ptr[1], ops_mul_2);

  // output initialization
  Tensor<std::float64_t> *temp_out_grad =
      new Tensor<std::float64_t>(*this->inputs[0]);
  ops_mul_2->initializeoutput(temp_out_grad);
  gradient_graph->addGradientNode(temp_out_grad);
  gradient_graph->addGradientEdge(ops_mul_2, temp_out_grad);
  this->outgoing_gradients.push_back(temp_out_grad);
  // End of d/dx[i] * z'
}

void Opssoftmax::compute() {
  unsigned inner_stride = 1;
  for (unsigned i = 0; i < this->axis; i++)
    inner_stride *= this->inputs[0]->getDimensions()[i];

  unsigned softmax_axis_len = this->inputs[0]->getDimensions()[this->axis];
  unsigned outer_stride = 1;
  for (unsigned i = this->axis + 1; i < this->inputs[0]->getNoOfDimensions();
       i++)
    outer_stride *= this->inputs[0]->getDimensions()[i];

  std::float64_t *ptr[2];
  ptr[0] = this->inputs[0]->getData();
  ptr[1] = this->output->getData();
  kernel_dispatch(ptr, this->axis, softmax_axis_len, inner_stride,
                  outer_stride);
}

void Opssoftmax::initializeinputs(Tensor<std::float64_t> **inputs) {
  this->inputs.push_back(inputs[0]);
}

void Opssoftmax::initializeoutput(Tensor<std::float64_t> *outputs) {
  this->output = outputs;
  *(this->output) = *(inputs[0]);
}

void Opssoftmax::printinputs() {
  unsigned i;
  for (i = 0; i < 1; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opssoftmax::printoutput() {
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}

Tensor<std::float64_t> *
Opssoftmax::getOutgoingGradientTensor(Tensor<std::float64_t> *gradient_input) {
  auto it = std::find(inputs.begin(), inputs.end(), gradient_input);
  Tensor<std::float64_t> *ptr = nullptr;
  if (inputs.end() != it) {
    int idx = std::distance(inputs.begin(), it);
    ptr = outgoing_gradients[idx];
  }
  return ptr;
}

Tensor<std::float64_t> *
Opssoftmax::getIncomingGradientTensor(Tensor<std::float64_t> *tensor) {
  return incoming_gradient;
}

void Opssoftmax::kernel_dispatch(std::float64_t *const *const ptr,
                                 unsigned const axis, unsigned const axis_len,
                                 unsigned const inner_stride,
                                 unsigned const outer_stride) {
  KernelType kernel = get_global_kernel();
#ifdef ENABLE_CUDA
  switch (kernel) {
  case KernelType::GPU: {
    double *d_arr[2];
    d_arr[0] = reinterpret_cast<double *>(ptr[0]);
    d_arr[1] = reinterpret_cast<double *>(ptr[1]);
    gpu::gpu_mat_softmax_f64(d_arr, axis, axis_len, inner_stride, outer_stride);
    break;
  }
  case KernelType::AVX2:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_softmax_f64(ptr, axis, axis_len, inner_stride, outer_stride);
    } else {
      throw std::runtime_error("AVX2 not supported on this CPU");
    }
    break;
  case KernelType::CPU_SCALAR:
    cpu::__msoftmax(ptr, axis, axis_len, inner_stride, outer_stride);
    break;
  case KernelType::AUTO:
  default: {
    double *d_arr[2];
    d_arr[0] = reinterpret_cast<double *>(ptr[0]);
    d_arr[1] = reinterpret_cast<double *>(ptr[1]);
    gpu::gpu_mat_softmax_f64(d_arr, axis, axis_len, inner_stride, outer_stride);
    break;
  }
  }
#else
  switch (kernel) {
  case KernelType::GPU:
    throw std::runtime_error("GPU kernel requested but CUDA not enabled");
  case KernelType::AVX2:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_softmax_f64(ptr, axis, axis_len, inner_stride, outer_stride);
    } else {
      throw std::runtime_error("AVX2 not supported on this CPU");
    }
    break;
  case KernelType::CPU_SCALAR:
    cpu::__msoftmax(ptr, axis, axis_len, inner_stride, outer_stride);
    break;
  case KernelType::AUTO:
  default:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_softmax_f64(ptr, axis, axis_len, inner_stride, outer_stride);
    } else {
      cpu::__msoftmax(ptr, axis, axis_len, inner_stride, outer_stride);
    }
    break;
  }
#endif
}
