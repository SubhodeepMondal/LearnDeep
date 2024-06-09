#pragma ONCE

class DenseForward
{
    NDMath math;
protected:
    void predict(NDArray<double, 0> input, NDArray<double, 0> weights, NDArray<double, 0> biases, NDArray<double, 0> output);

    void fit(NDArray<double, 1> input_gpu, NDArray<double, 1> weights_gpu, NDArray<double, 1> biases_gpu, NDArray<double, 1> output_gpu, cudaStream_t stream);
};

class BatchNormalizationForward
{
    NDMath math;
    NDArray<double, 1> mean;
    NDArray<double, 1> stdDiv;
    NDArray<double, 1> stdDivTemp;
    NDArray<double, 1> moving_mean;
    NDArray<double, 1> moving_stdDiv;
    double alpha = 0.9;
    unsigned isMeanStdDivInitilized = 0;

    void findMean(NDArray<double, 1> input_gpu, cudaStream_t stream);

    void findStdDiv(NDArray<double, 1> input_gpu, cudaStream_t stream);

    void movingMean(cudaStream_t stream);

    void movingStdDiv(cudaStream_t stream);

    void normalize(NDArray<double, 1> input_gpu, NDArray<double, 1> output_gpu, cudaStream_t stream);

    void scale(NDArray<double, 1> gamma_gpu, NDArray<double, 1> beta_gpu, NDArray<double, 1> output_gpu, cudaStream_t stream);

protected:
    void predict(NDArray<double, 0> input, NDArray<double, 0> gamma, NDArray<double, 0> beta, NDArray<double, 0> output);
    
    void fit(NDArray<double, 1> input_gpu, NDArray<double, 1> gamma_gpu, NDArray<double, 1> beta_gpu, NDArray<double, 1> output_gpu, cudaStream_t stream);
};