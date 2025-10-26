#include "opskernel.h"
#ifdef CUDA_ENABLED
#include <LAS/gpu_interface.cuh>
#endif

#include <LAS/CPULibrary.h>
#include <LAS/avx2_micro_kernels.h>
#include <framework/MathLibrary.h>
#include <kernel/opskernel.h>

void Opsscale::recursive_iterator(unsigned index, unsigned *dimension_arr,
                                  std::string function_name, unsigned *ui_arr,
                                  std::float64_t *dl_arr,
                                  Tensor<std::float64_t> *misc_arr) {
  if (index < 2) {
    unsigned i, inpA_x, inpA_y, inpB_x, inpB_y, out_x, out_y;
    unsigned a_plane_size, b_plane_size, c_plane_size, a_index, b_index,
        c_index;

    inpA_x = (inputs[0]->getNoOfDimensions() > 0)
                 ? inputs[0]->getDimensions()[0]
                 : 1;
    inpA_y = (inputs[0]->getNoOfDimensions() > 1)
                 ? inputs[0]->getDimensions()[1]
                 : 1;

    out_x = (output->getNoOfDimensions() > 0) ? output->getDimensions()[0] : 1;
    out_y = (output->getNoOfDimensions() > 1) ? output->getDimensions()[1] : 1;

    a_plane_size = inpA_x * inpA_y;
    c_plane_size = out_x * out_y;

    a_index = b_index = c_index = 0;
    if (inputs[0]->getNoOfDimensions() > 2)
      for (i = 2; i < inputs[0]->getNoOfDimensions(); i++) {
        a_index += a_plane_size * dimension_arr[i];
        c_index += c_plane_size * dimension_arr[i];

        a_plane_size *= inputs[0]->getDimensions()[i];
        c_plane_size *= output->getDimensions()[i];
      }
    unsigned a[2];
    std::float64_t *ptr[3];

    a[0] = inpA_x;
    a[1] = inpA_y;

    ptr[0] = inputs[0]->getData() + a_index;
    ptr[1] = dl_arr;
    ptr[2] = output->getData() + c_index;

    kernel_dispatch(ptr, a);
  } else {
    for (unsigned i = 0; i < inputs[0]->getDimensions()[index]; i++) {
      dimension_arr[index] = i;
      recursive_iterator(index - 1, dimension_arr, function_name, ui_arr,
                         dl_arr, misc_arr);
    }
  }
};

void Opsscale::compute() {
  unsigned *arr;

  arr = new unsigned[inputs[0]->getNoOfDimensions()];

  recursive_iterator(inputs[0]->getNoOfDimensions() - 1, arr,
                     "matrix_scaler_multiplication", NULL, scale_factor, NULL);
  delete[] arr;
}

void Opsscale::addGradGraph(Graph *gradient_graph) {
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

  // graph setup for d/dx[i] * z'
  Ops *ops_scale = new Opsscale;

  // input initialization
  ops_scale->initializeinputs(&this->incoming_gradient);
  ops_scale->initializeScale(this->scale_factor[0]);
  gradient_graph->addGradientNode(ops_scale);
  gradient_graph->addGradientNode(this->incoming_gradient);
  gradient_graph->addGradientEdge(this->incoming_gradient, ops_scale);

  // output initialization
  this->outgoing_gradients[0] = new Tensor<std::float64_t>(*this->inputs[0]);
  ops_scale->initializeoutput(this->outgoing_gradients[0]);
  gradient_graph->addGradientNode(this->outgoing_gradients[0]);
  gradient_graph->addGradientEdge(ops_scale, this->outgoing_gradients[0]);
  // End of d/dx[i] * z'
}

void Opsscale::initializeinputs(Tensor<std::float64_t> **inputs) {
  this->inputs.push_back(inputs[0]);
}

void Opsscale::initializeoutput(Tensor<std::float64_t> *outputs) {
  this->output = outputs;
  // *(this->output) = *(inputs[0]);
  this->output->reshape(this->inputs[0]->getNoOfDimensions(),
                        this->inputs[0]->getDimensions());
}

void Opsscale::printinputs() {
  unsigned i;
  for (i = 0; i < 1; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opsscale::printoutput() {
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}

Tensor<std::float64_t> *
Opsscale::getOutgoingGradientTensor(Tensor<std::float64_t> *gradient_input) {

  if (this->inputs[0] == gradient_input) {
    // LOG(INFO) << "Requested gradint for the tensor found.\n";
    return this->outgoing_gradients[0];
  } else {
    // LOG(FATAL) << "Requested gradint for the tensor doesn't exist.\n";
    return NULL;
  }
}
std::vector<Tensor<std::float64_t> *>
Opsscale::getAllOutgoingGradientTensors() {
  std::vector<Tensor<std::float64_t> *> grads;
  grads.push_back(outgoing_gradients[0]);
  return grads;
}
void Opsscale::kernel_dispatch(std::float64_t **ptr, unsigned *arr) {
#ifdef CUDA_ENABLED
  double *d_arr[3];
  d_arr[0] = reinterpret_cast<double *>(ptr[0]);
  d_arr[1] = reinterpret_cast<double *>(ptr[1]);
  d_arr[2] = reinterpret_cast<double *>(ptr[2]);

  gpu::gpu_mat_scale_f64(d_arr, arr);
#else
  if (__builtin_cpu_supports("avx2")) {
    avx2::avx2_scale_f64(ptr, arr);
  } else {
    cpu::__mscalermul(ptr, arr);
  }
#endif
}
