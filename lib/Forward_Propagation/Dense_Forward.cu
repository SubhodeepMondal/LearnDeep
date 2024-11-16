#include "NDynamicArray.h"
#include "MathLibrary.h"
#include "Forward_Propagation.h"

void DenseForward::predict(NDArray<double, 0> input, NDArray<double, 0> weights, NDArray<double, 0> biases, NDArray<double, 0> output)
{
    math.matrixDotMultiplication(input, weights, biases, output);
}
void DenseForward::fit(NDArray<double, 1> input_gpu, NDArray<double, 1> weights_gpu, NDArray<double, 1> biases_gpu, NDArray<double, 1> output_gpu, cudaStream_t stream)
{
    // std::cout << "In fit\n";
    // std::cout << "\nInput\n";
    // std::cout << "input ptr: " << input_gpu.getData() << "\n";
    // input_gpu.printData();
    // std::cout << "\nweights & Biases\n";
    // weights_gpu.printData();
    // std::cout << " ";
    // biases_gpu.printData();

    
    math.matrixDotMultiplication(input_gpu, weights_gpu, biases_gpu, output_gpu, stream);

    // std::cout << "output\n";
    // output_gpu.printData();
}
