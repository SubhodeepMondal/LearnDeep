#include "opskernel.h"
#ifdef ENABLE_CUDA
#include <core/LAS/gpu_interface.cuh>
#endif

// Library Headers
#include "kernelmanager.h"
#include "opskernel.h"
#include <core/LAS/CPULibrary.h>
#include <core/LAS/avx2_micro_kernels.h>
#include <core/framework/MathLibrary.h>

void Opslog::addGradGraph(Graph *gradient_graph) {
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

  Tensor<std::float64_t> *temp_grad_tensors =
      new Tensor<std::float64_t>(*this->inputs[0]);
  temp_grad_tensors->initData(1.0);

  Ops *ops_div = new Opsdiv();

  tensor_ptr[0] = temp_grad_tensors;
  tensor_ptr[1] = this->inputs[0];

  ops_div->initializeinputs(tensor_ptr);
  gradient_graph->addGradientNode(ops_div);
  gradient_graph->addGradientNode(tensor_ptr[0]);
  gradient_graph->addGradientNode(tensor_ptr[1]);
  gradient_graph->addGradientEdge(tensor_ptr[0], ops_div);
  gradient_graph->addGradientEdge(tensor_ptr[1], ops_div);

  Tensor<std::float64_t> *temp_output =
      new Tensor<std::float64_t>(*this->inputs[0]);
  ops_div->initializeoutput(temp_output);
  gradient_graph->addGradientNode(temp_output);
  gradient_graph->addGradientEdge(ops_div, temp_output);

  // graph setup for d/dx[i] * z'
  Ops *ops_mul = new Opsmul;
  tensor_ptr[0] = temp_output;
  tensor_ptr[1] = this->incoming_gradient;

  // input initialization
  ops_mul->initializeinputs(tensor_ptr);
  gradient_graph->addGradientNode(ops_mul);
  gradient_graph->addGradientNode(tensor_ptr[0]);
  gradient_graph->addGradientNode(tensor_ptr[1]);
  gradient_graph->addGradientEdge(tensor_ptr[0], ops_mul);
  gradient_graph->addGradientEdge(tensor_ptr[1], ops_mul);

  // output initialization
  this->outgoing_gradients.push_back(
      new Tensor<std::float64_t>(*this->inputs[0]));
  ops_mul->initializeoutput(this->outgoing_gradients[0]);
  gradient_graph->addGradientNode(this->outgoing_gradients[0]);
  gradient_graph->addGradientEdge(ops_mul, this->outgoing_gradients[0]);
  // End of d/dx[i] * z'
}

void Opslog::compute() {
  unsigned *arr = new unsigned[this->inputs[0]->getNoOfDimensions() + 1];

  std::float64_t *ptr[2];

  ptr[0] = inputs[0]->getData();
  ptr[1] = output->getData();

  arr[0] = this->inputs[0]->getNoOfDimensions();
  for (unsigned i = 0; i < arr[0]; i++)
    arr[i + 1] = this->inputs[0]->getDimensions()[i];

  kernel_dispatch(ptr, arr);
  delete[] arr;
}

void Opslog::initializeinputs(Tensor<std::float64_t> **inputs) {
  this->inputs.push_back(inputs[0]);
}

void Opslog::initializeoutput(Tensor<std::float64_t> *outputs) {
  this->output = outputs;
  // *(this->output) = *(inputs[0]);
  this->output->reshape(this->inputs[0]->getNoOfDimensions(),
                        this->inputs[0]->getDimensions());
}

void Opslog::printinputs() {
  unsigned i;
  for (i = 0; i < 1; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opslog::printoutput() {
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}

Tensor<std::float64_t> *
Opslog::getOutgoingGradientTensor(Tensor<std::float64_t> *gradient_input) {
  int i, it;
  bool flag = false;
  for (i = 0; i < this->inputs.size(); i++)
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

void Opslog::kernel_dispatch(std::float64_t **ptr, unsigned *arr) {
  KernelType kernel = get_global_kernel();
#ifdef ENABLE_CUDA
  switch (kernel) {
  case KernelType::GPU: {
    double *d_arr[3];
    d_arr[0] = reinterpret_cast<double *>(ptr[0]);
    d_arr[1] = reinterpret_cast<double *>(ptr[1]);
    d_arr[2] = reinterpret_cast<double *>(ptr[2]);
    gpu::gpu_log_f64(d_arr, arr);
    break;
  }
  case KernelType::AVX2:
    if (!this->warning_handled) {
      std::cerr << "AVX2 kernel requested but, log has no AVX2 "
                   "implementation, falling back to CPU kernel\n";
      this->warning_handled = true;
    }
  case KernelType::CPU_SCALAR:
  case KernelType::AUTO:
  default:
    cpu::__mlog(ptr, arr);
    break;
  }
#else
  switch (kernel) {
  case KernelType::GPU:
    throw std::runtime_error("GPU kernel requested but CUDA not enabled");
  case KernelType::AVX2:
    if (!this->warning_handled) {
      std::cerr << "AVX2 kernel requested but, log has no AVX2 "
                   "implementation, falling back to CPU kernel\n";
      this->warning_handled = true;
    }
  case KernelType::CPU_SCALAR:
    cpu::__mlog(ptr, arr);
    break;
  case KernelType::AUTO:
  default:
    cpu::__mlog(ptr, arr);
    break;
  }
#endif
}
