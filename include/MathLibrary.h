#pragma ONCE

#include "NDynamicArray.h"
#include "CPULibrary.h"

template <typename T, int typeFlag>
class NDMath : public NDArray<T, typeFlag>
{
    struct arg_list
    {
        unsigned value;
        struct arg_list *next;
    };
    struct arg_list *head, *ptr, *ptr_prev;
    unsigned *arr_dims;

    void recursive_sum(unsigned, unsigned, unsigned *, NDMath<T, typeFlag>, T *, NDMath<T, typeFlag>);

public:
    NDMath(const NDMath<T, typeFlag> &ndmath) : NDArray<T, typeFlag>(ndmath) {}

    template <typename... args>
    NDMath(unsigned num, args... Args) : NDArray<T, typeFlag>(num, Args...) {}

    NDMath() {}

    ~NDMath() {}

    NDMath<T, typeFlag> &operator=(const NDMath<T, typeFlag> &ndmath)
    {
        NDArray<T, typeFlag>::operator=(ndmath);
        return *this;
    }

    void recursive_iterator(unsigned index, unsigned *dimension_arr, NDArray<T, typeFlag> input, NDArray<T, typeFlag> &output)
    {
        if (index < 2)
        {
            unsigned i, inpA_x, inpA_y, inpB_x, inpB_y, out_x, out_y;
            unsigned a_plane_size, b_plane_size, c_plane_size, a_index, b_index, c_index;

            inpA_x = NDMath<T, typeFlag>::getDimensions()[0];
            inpA_y = NDMath<T, typeFlag>::getDimensions()[1];

            inpB_x = input.getDimensions()[0];
            inpB_y = input.getDimensions()[1];
            
            out_x = output.getDimensions()[0];
            out_y = output.getDimensions()[1];

            a_plane_size = inpA_x * inpA_y;
            b_plane_size = inpB_x * inpB_y;
            c_plane_size = out_x * out_y;

            a_index = b_index = c_index = 0;
            // std::cout << "a index: " << a_plane_size << " b index: " << b_plane_size << " c index: " << c_plane_size << "\n";
            if (input.getNoOfDimensions() > 2)
                for (i = 2; i < input.getNoOfDimensions(); i++)
                {
                    std::cout << dimension_arr[i] << " ";
                    a_index += a_plane_size * dimension_arr[i];
                    b_index += b_plane_size * dimension_arr[i];
                    c_index += c_plane_size * dimension_arr[i];

                    a_plane_size *= this->getDimensions()[i];
                    b_plane_size *= input.getDimensions()[i];
                    c_plane_size *= output.getDimensions()[i];
                    std::cout << "a index: " << a_index << " b index: " << b_index << " c index: " << c_index << "\n";
                }

            // std::cout << "a index: " << a_index << " b index: " << b_index << " c index: " << c_index << "\n";
            // std::cout << "a ptr: " << this->getData()[0] << " b index: " << input.getData()[0] << " c index: " << output.getData()[0] << "\n";

            // std::cout << "a index: " << inpA_x << " b index: " << inpA_y << " c index: " << inpB_x << "\n";
            cpu::__mmul(NDMath<T, typeFlag>::getData() + a_index, input.getData() + b_index, output.getData() + c_index, inpB_x, inpB_y, inpA_y);

            // cpu::__mmulconventional(NDMath<T, typeFlag>::getData() + a_index, input.getData() + b_index, output.getData() + c_index, inpB_x, inpB_y, inpA_y);

        }
        else
        {
            for (unsigned i = 0; i < NDArray<T, typeFlag>::getDimensions()[index]; i++)
            {
                dimension_arr[index] = i;
                recursive_iterator(index - 1, dimension_arr, input, output);
            }
        }
    }

    NDMath<T, typeFlag> matrixMultiplication(const NDMath<double, 0>);

    NDMath<T, typeFlag> matrixMultiplication(const double);

    NDMath<T, typeFlag> matrixAddition(const NDMath<double, 0>);

    NDMath<T, typeFlag> operator*(const NDMath<double, 0>);

    NDMath<T, typeFlag> operator+(const NDMath<double, 0>);

    NDMath<T, typeFlag> operator-(const NDMath<double, 0>);

    NDMath<T, typeFlag> matrixVectorAddition(const NDMath<double, 0>);

    void matrixTranspose();

    void reducesum(NDMath<T, typeFlag> *);

    template <typename first_dim, typename... Args>
    void reducesum(NDMath<T, typeFlag> *, first_dim, Args...);

    template <typename... Args>
    NDMath<T, typeFlag> sum(Args... args);

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

#include "../lib/Math/MathLibrary.cpp"