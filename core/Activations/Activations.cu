#include<map>
#include "NDynamicArray.h"
#include "MathLibrary.h"
#include "Activations.h"

void Relu_Activation::activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream)
{
    math.reluActivation(output, delta_activation, stream);
}
void Relu_Activation::activate(NDArray<double, 0> output)
{
    math.reluActivation(output);
}

void Sigmoid_Activation::activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream)
{
    math.sigmoidActivation(output, delta_activation, stream);
}
void Sigmoid_Activation::activate(NDArray<double, 0> output)
{
    math.sigmoidActivation(output);
}

void Linear_Activation::activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream)
{
    math.linearActivation(output, delta_activation, stream);
}
void Linear_Activation::activate(NDArray<double, 0> output)
{
}

void Softmax_Activation::activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream)
{
    if (!flag)
    {
        softmax_sum = NDArray<double, 1>(1, output.getDimensions()[1]);
        flag = 1;
    }
    math.softmaxActivation(output, softmax_sum, delta_activation, stream);
}

void Softmax_Activation::activate(NDArray<double, 0> output)
{
}
