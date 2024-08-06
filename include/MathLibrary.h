#pragma ONCE

#include "NDynamicArray.h"
#include "CPULibrary.h"

template <typename T, int typeFlag>
class NDMath : public NDArray<T, typeFlag>
{

public:
    template <typename... args>
    NDMath(args... Args) : NDArray<T, typeFlag>(Args...) {}
    NDMath() {};

    NDMath<T, typeFlag> matrixMultiplication(NDMath<double, 0> input)
    {
        NDMath<T, typeFlag> output;
        unsigned i, j, plane_dimension, no_of_dimensions, actual_index, flag = 1;
        unsigned dim_x, dim_y, dim_z;

        no_of_dimensions = NDMath<T, typeFlag>::getNoOfDimensions();

        dim_x = NDMath<T, typeFlag>::getDimensions()[1];
        dim_y = NDMath<T, typeFlag>::getDimensions()[0];
        dim_z = input.getDimensions()[0];

        if (no_of_dimensions == input.getNoOfDimensions())
        {
            for (i = 2; i < no_of_dimensions; i++)
                if (NDMath<T, typeFlag>::getDimensions()[i] != input.getDimensions()[i])
                {
                    flag = 0;
                    break;
                }
            if (flag && this->getDimensions()[0] == input.getDimensions()[1])
            {
                // output = NDMath<T, typeFlag>(no_of_dimensions, NDMath<T, typeFlag>::getDimensions());

                output = NDMath<T, typeFlag>(dim_y, dim_z);
                if (no_of_dimensions < 3)
                {
                    cpu::__mmul(NDMath<T, typeFlag>::getData(), input.getData(), output.getData(), dim_x, dim_z, dim_y);
                }
                else
                {
                    plane_dimension = dim_x * dim_y;
                    actual_index = 0;
                    for (i = 2; i < no_of_dimensions; i++)
                    {
                        for (j = 0; j < NDMath<T, typeFlag>::getDimensions()[i]; j++)
                        {
                            cpu::__mmul(NDMath<T, typeFlag>::getData() + actual_index, input.getData() + actual_index, output.getData() + actual_index, dim_x, dim_z, dim_y);
                            actual_index += plane_dimension;
                        }
                    }
                }
            }
        }
        else
        {
            return NULL;
        }

        return output;
    }

    // NDArray<double, 0> multiplication(NDArray<double, 0>, NDArray<double, 0>, int );
    // void matrixDotMultiplication(NDArray<double, 0> input, NDArray<double, 0> weights, NDArray<double, 0> biases, NDArray<double, 0> output);
    // void matrixDotMultiplication(NDArray<double, 1> input, NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> output, cudaStream_t stream);
    // void updateLearningRateWeightsAdagrad(NDArray<double, 1> epsalon, NDArray<double, 1> sum_delta_weights, NDArray<double, 1> delta_weights, NDArray<double, 1> learning_rate_weights, cudaStream_t stream);
    // void updateLearningRateBiasesAdagrad(NDArray<double, 1> epsalon, NDArray<double, 1> sum_delta_biases, NDArray<double, 1> delta_biases, NDArray<double, 1> learning_rate_biases, cudaStream_t stream);
    // void updateLearningRateWeightsAdadelta(NDArray<double, 1> epsalon, NDArray<double, 1> sigma, NDArray<double, 1> sum_delta_weights, NDArray<double, 1> delta_weights, NDArray<double, 1> learning_rate_weights, cudaStream_t stream);
    // void updateLearningRateBiasesAdadelta(NDArray<double, 1> epsalon, NDArray<double, 1> sigma, NDArray<double, 1> sum_delta_biases, NDArray<double, 1> delta_biases, NDArray<double, 1> learning_rate_biases, cudaStream_t stream);
    // void updateWeights(NDArray<double, 1> weights, NDArray<double, 1> learning_rate, NDArray<double, 1> delta_weights, cudaStream_t stream)
    // void updateBiases(NDArray<double, 1> biases, NDArray<double, 1> learning_rate, NDArray<double, 1> delta_biases, cudaStream_t stream);
    // void updateWeightsSGDmomentum(NDArray<double, 1> sigma, NDArray<double, 1> weights, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_weights, NDArray<double, 1> delta_weights, cudaStream_t stream);
    // void updateBiasesSGDmomentum(NDArray<double, 1> sigma, NDArray<double, 1> biases, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_biases, NDArray<double, 1> delta_biases, cudaStream_t stream);
    // void updateWeightsRMSpropDense(NDArray<double, 1> sigma, NDArray<double, 1> epsalon, NDArray<double, 1> weights, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_weights, NDArray<double, 1> delta_weights, cudaStream_t stream);
    // void updateBiasesRMSpropDense(NDArray<double, 1> sigma, NDArray<double, 1> epsalon, NDArray<double, 1> biases, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_biases, NDArray<double, 1> delta_biases, cudaStream_t stream);
    // void updateWeightsADAMDense(NDArray<double, 1> sigma, NDArray<double, 1> epsalon, NDArray<double, 1> weights, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_weights, NDArray<double, 1> sum_delta_weights_square, NDArray<double, 1> delta_weights, cudaStream_t stream);
    // void updateBiasesADAMDense(NDArray<double, 1> sigma, NDArray<double, 1> epsalon, NDArray<double, 1> biases, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_biases, NDArray<double, 1> sum_delta_biases_squared, NDArray<double, 1> delta_biases, cudaStream_t stream);
    // void getDifferentialWeights(NDArray<double, 1> input, NDArray<double, 1> delta_output, NDArray<double, 1> difference, NDArray<double, 1> delta_weights, NDArray<double, 1> delta_weights_intermediate, cudaStream_t stream);
    // void getDifferentialBiases(NDArray<double, 1> delta_output, NDArray<double, 1> difference, NDArray<double, 1> delta_biases, NDArray<double, 1> delta_biases_intermediate, cudaStream_t stream);
    // void getDifferentialInput(NDArray<double, 1> weights, NDArray<double, 1> delta_output, NDArray<double, 1> difference, NDArray<double, 1> difference_input, NDArray<double, 1> delta_input_intermediate, NDArray<double, 1> delta_input, cudaStream_t stream);
    // void reluActivation(NDArray<double, 1> input, NDArray<double, 1> d_activation, cudaStream_t stream);
    // void reluActivation(NDArray<double, 0> input);
    // void sigmoidActivation(NDArray<double, 1> input, NDArray<double, 1> d_activation, cudaStream_t stream);
    // void sigmoidActivation(NDArray<double, 0> input);
    // void linearActivation(NDArray<double, 1> input, NDArray<double, 1> d_activation, cudaStream_t stream);
    // void softmaxActivation(NDArray<double, 1> input, NDArray<double, 1> softmax_sum, NDArray<double, 1> d_activation, cudaStream_t stream);
    // void squaredError(NDArray<double, 1> Difference, NDArray<double, 1> Squared_Error, cudaStream_t stream);
    // void findMean(NDArray<double, 1> X, NDArray<double, 1> Y, cudaStream_t stream);
    // double findMean(NDArray<double, 0> A);
    // void argMax(NDArray<double, 1> probabilities, NDArray<double, 1> one_hot_code, cudaStream_t stream);
    // NDArray<double, 0> findSquare(NDArray<double, 0> A);
    // NDArray<double, 0> findSquareRoot(NDArray<double, 0> A);
    // void findDifference(NDArray<double, 1> Y_predict, NDArray<double, 1> Y_target, NDArray<double, 1> Difference, cudaStream_t stream);
    // NDArray<double, 0> findDifference(NDArray<double, 0> A, NDArray<double, 0> B);
    // void binaryCrossEntropy(NDArray<double, 1> Y_predict, NDArray<double, 1> Y_target, NDArray<double, 1> Difference, NDArray<double, 1> Cost, cudaStream_t stream);
    // void confusionMatrix(NDArray<double, 1> predict, NDArray<double, 1> actual, NDArray<double, 1> confusion_matrix, cudaStream_t stream);
    // void accuracyValue(NDArray<double, 1> confusion_matrix, NDArray<double, 1> accuracy_value, unsigned no_of_classes, unsigned no_of_samples, cudaStream_t stream);
};

// #include "../lib/Math/MathLibrary.cu"