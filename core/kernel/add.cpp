#ifdef CUDA_ENABLED
#include <LAS/gpu_interface.cuh>
#endif

#include <LAS/CPULibrary.h>
#include <LAS/avx2_micro_kernels.h>

#include <framework/MathLibrary.h>
#include <kernel/opskernel.h>

void Opsadd::recursive_iterator(unsigned index, unsigned *dimension_arr,
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

    inpB_x = (inputs[1]->getNoOfDimensions() > 0)
                 ? inputs[1]->getDimensions()[0]
                 : 1;
    inpB_y = (inputs[1]->getNoOfDimensions() > 1)
                 ? inputs[1]->getDimensions()[1]
                 : 1;

    out_x = (output->getNoOfDimensions() > 0) ? output->getDimensions()[0] : 1;
    out_y = (output->getNoOfDimensions() > 1) ? output->getDimensions()[1] : 1;

    a_plane_size = inpA_x * inpA_y;
    b_plane_size = inpB_x * inpB_y;
    c_plane_size = out_x * out_y;

    a_index = b_index = c_index = 0;
    if (inputs[1]->getNoOfDimensions() > 2)
      for (i = 2; i < inputs[1]->getNoOfDimensions(); i++) {
        a_index += a_plane_size * dimension_arr[i];
        b_index += b_plane_size * dimension_arr[i];
        c_index += c_plane_size * dimension_arr[i];

        a_plane_size *= inputs[0]->getDimensions()[i];
        b_plane_size *= inputs[1]->getDimensions()[i];
        c_plane_size *= output->getDimensions()[i];
      }
    /* code */
    unsigned a[2];
    std::float64_t *ptr[3];

    a[0] = inpA_x;
    a[1] = inpA_y;

    ptr[0] = inputs[0]->getData() + a_index;
    ptr[1] = inputs[1]->getData() + b_index;
    ptr[2] = output->getData() + c_index;

    kernel_dispatch(ptr, a);

  } else {
    for (unsigned i = 0; i < inputs[0]->getDimensions()[index]; i++) {
      dimension_arr[index] = i;
      recursive_iterator(index - 1, dimension_arr, function_name, NULL, NULL,
                         NULL);
    }
  }
};

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

  std::vector<Tensor<std::float64_t> *> incoming_gradient =
      gradient_graph->getGradient(this);
  Tensor<std::float64_t> *tensor_ptr[2];

  // graph setup for accumulating incoming gradients y' = sum ( z' )
  if (incoming_gradient.size()) {
    Tensor<std::float64_t> *intermediate_gradient_sum;

    intermediate_gradient_sum = new Tensor<std::float64_t>(*this->output);
    intermediate_gradient_sum->initData(0.0);
    int i = 0;
    for (Tensor<std::float64_t> *inc_grad_tensor : incoming_gradient) {

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
    this->outgoing_gradients[i] = this->incoming_gradient;
}

void Opsadd::compute() {
  unsigned dim_x, dim_y;
  dim_x = inputs[0]->getDimensions()[0];
  dim_y = inputs[1]->getDimensions()[1];

  unsigned *arr = new unsigned[inputs[0]->getNoOfDimensions()];

  recursive_iterator(inputs[0]->getNoOfDimensions() - 1, arr, "matrix_addition",
                     NULL, NULL, NULL);

  delete[] arr;
}

void Opsadd::initializeinputs(Tensor<std::float64_t> **inputs) {
  unsigned i;
  this->no_of_inputs = 2;
  for (i = 0; i < this->no_of_inputs; i++) {
    this->inputs.push_back(inputs[i]);
  }
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
  for (i = 0; i < this->no_of_inputs; i++) {
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

Tensor<std::float64_t> *
Opsadd::getIncomingGradientTensor(Tensor<std::float64_t> *tensor) {
  return incoming_gradient;
}

void Opsadd::kernel_dispatch(std::float64_t **ptr, unsigned *arr) {

#ifdef CUDA_ENABLED
  double *d_arr[3];
  d_arr[0] = reinterpret_cast<double *>(ptr[0]);
  d_arr[1] = reinterpret_cast<double *>(ptr[1]);
  d_arr[2] = reinterpret_cast<double *>(ptr[2]);

  gpu::gpu_mat_add_f64(d_arr, arr);
#else
  if (__builtin_cpu_supports("avx2")) {
    avx2::avx2_add_f64(ptr, arr);
  } else {
    cpu::__madd(ptr, arr);
  }
#endif
}
