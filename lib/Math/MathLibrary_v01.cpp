    NDArray<double, 0> NDMath::multiplication(NDArray<double, 0> a, NDArray<double, 0> b, int gpu = 1)
    {
        int noDevice;
        int i, j, a_m, a_n, b_m, b_n;
        dim3 block, grid;
        double *ptrA, *ptrB, *ptrC;

        a_m = a.getDimensions()[0];
        a_n = a.getDimensions()[1];
        b_m = b.getDimensions()[0];
        b_n = b.getDimensions()[1];

        NDArray<double, 0> c(2, a_m, b_n);
        NDArray<double, 0> e(2, a_m, b_n);

        cudaGetDeviceCount(&noDevice);

        if (a_n == b_m)
        {
            if (noDevice > 0)
            {

                cudaMalloc((double **)&ptrA, sizeof(double) * a.getNoOfElem());
                cudaMalloc((double **)&ptrB, sizeof(double) * b.getNoOfElem());
                cudaMalloc((double **)&ptrC, sizeof(double) * c.getNoOfElem());

                cudaMemcpy(ptrA, a.getData(), sizeof(double) * a.getNoOfElem(), cudaMemcpyHostToDevice);
                cudaMemcpy(ptrB, b.getData(), sizeof(double) * b.getNoOfElem(), cudaMemcpyHostToDevice);

                block.x = 32;
                block.y = 32;
                grid.x = 1;
                grid.y = ceil(a_m / 32.0f);

                cudaSetDevice(0);

                for (i = 0; i < b_m; i += 32)
                {
                    for (j = 0; j < b_n; j += 32)
                    {
                        gpu::cudaMatrixMul<<<grid, block>>>(ptrA, ptrB, ptrC, a_m, a_n, b_m, b_n, i, j);
                    }
                }
                cudaDeviceSynchronize();

                cudaMemcpy(c.getData(), ptrC, sizeof(double) * c.getNoOfElem(), cudaMemcpyDeviceToHost);

                cudaDeviceSynchronize();

                cudaFree(ptrA);
                cudaFree(ptrB);
                cudaFree(ptrC);
            }
            else
            {
                double *ptrD;
                ptrD = new double[a_m * b_n];
                // cpu::matrixMul(ptrA, ptrB, ptrD, a_m, a_n, b_m, b_n);
                c.initData(ptrD);
            }
        }
        return c;
    }

    void NDMath::matrixDotMultiplication(NDArray<double, 0> input, NDArray<double, 0> weights, NDArray<double, 0> biases, NDArray<double, 0> output)
    {
        // unsigned intr_x, intr_y;

        // intr_x = weights.getDimensions()[0];
        // intr_y = weights.getDimensions()[1];

        // cpu::matrixDotMul(input.getData(), weights.getData(), biases.getData(), output.getData(), intr_x, intr_y);
    }

    void NDMath::matrixDotMultiplication(NDArray<double, 1> input, NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> output, cudaStream_t stream)
    {
        unsigned intr_x, intr_y, intr_z;
        dim3 block, grid;

        intr_x = weights.getDimensions()[0];     // no of neurons
        intr_y = input.getDimensions()[1];       // no of batches
        intr_z = weights.getDimensions()[1] + 1; // no of features

        if (!isIntermediateOutputUpdated)
        {
            intermediate_output = NDArray<double, 1>(3, intr_x, intr_y, intr_z);
            isIntermediateOutputUpdated = 1;
        }

        block.x = (intr_x > 8) ? 8 : intr_x;
        block.y = (intr_y > 8) ? 8 : intr_y;
        block.z = (intr_z > 16) ? 16 : intr_z;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);
        grid.z = ceil((float)intr_z / block.z);

        gpu::matrixDotMul<<<grid, block, 0, stream>>>(input.getData(), weights.getData(), biases.getData(), intermediate_output.getData(), intr_x, intr_y, intr_z - 1);

        block.z = 1;
        grid.z = 1;

        gpu::matrixRollingSum<<<grid, block, 0, stream>>>(intermediate_output.getData(), output.getData(), intr_x, intr_y, intr_z);
    }

    void NDMath::updateLearningRateWeightsAdagrad(NDArray<double, 1> epsalon, NDArray<double, 1> sum_delta_weights, NDArray<double, 1> delta_weights, NDArray<double, 1> learning_rate_weights, cudaStream_t stream)
    {
        // unsigned intr_x, intr_y;
        // dim3 grid, block;

        // intr_x = delta_weights.getDimensions()[0];
        // intr_y = delta_weights.getDimensions()[1];

        // block.x = (intr_x > 32) ? 32 : intr_x;
        // block.y = (intr_y > 32) ? 32 : intr_y;

        // grid.x = ceil((float)intr_x / block.x);
        // grid.y = ceil((float)intr_y / block.y);

        // gpu::matrixUpdateLearningRateAdagrad<<<grid, block, 0, stream>>>(*epsalon.getData(), NULL, delta_weights.getData(), sum_delta_weights.getData(), learning_rate_weights.getData(), intr_x, intr_y, 1);
    }

    void NDMath::updateLearningRateBiasesAdagrad(NDArray<double, 1> epsalon, NDArray<double, 1> sum_delta_biases, NDArray<double, 1> delta_biases, NDArray<double, 1> learning_rate_biases, cudaStream_t stream)
    {
        // unsigned intr_x;
        // dim3 grid, block;

        // intr_x = delta_biases.getDimensions()[0];

        // block.x = (intr_x > 32) ? 32 : intr_x;

        // grid.x = ceil((float)intr_x / block.x);

        // gpu::matrixUpdateLearningRateAdagrad<<<grid, block, 0, stream>>>(*epsalon.getData(), NULL, delta_biases.getData(), sum_delta_biases.getData(), learning_rate_biases.getData(), intr_x, 1, 1);
    }

    void NDMath::updateLearningRateWeightsAdadelta(NDArray<double, 1> epsalon, NDArray<double, 1> sigma, NDArray<double, 1> sum_delta_weights, NDArray<double, 1> delta_weights, NDArray<double, 1> learning_rate_weights, cudaStream_t stream)
    {
        // unsigned intr_x, intr_y;
        // dim3 grid, block;

        // intr_x = delta_weights.getDimensions()[0];
        // intr_y = delta_weights.getDimensions()[1];

        // block.x = (intr_x > 32) ? 32 : intr_x;
        // block.y = (intr_y > 32) ? 32 : intr_y;

        // grid.x = ceil((float)intr_x / block.x);
        // grid.y = ceil((float)intr_y / block.y);

        // gpu::matrixUpdateLearningRateAdadelta<<<grid, block, 0, stream>>>(*epsalon.getData(), *sigma.getData(), delta_weights.getData(), sum_delta_weights.getData(), learning_rate_weights.getData(), NULL,intr_x, intr_y, 1);
    }

    void NDMath::updateLearningRateBiasesAdadelta(NDArray<double, 1> epsalon, NDArray<double, 1> sigma, NDArray<double, 1> sum_delta_biases, NDArray<double, 1> delta_biases, NDArray<double, 1> learning_rate_biases, cudaStream_t stream)
    {
        // unsigned intr_x;
        // dim3 grid, block;

        // intr_x = delta_biases.getDimensions()[0];

        // block.x = (intr_x > 32) ? 32 : intr_x;

        // grid.x = ceil((float)intr_x / block.x);

        // gpu::matrixUpdateLearningRateAdadelta<<<grid, block, 0, stream>>>(*epsalon.getData(), *sigma.getData(), delta_biases.getData(), sum_delta_biases.getData(), learning_rate_biases.getData(),NULL, intr_x, 1, 1);
    }

    void NDMath::updateWeights(NDArray<double, 1> weights, NDArray<double, 1> learning_rate, NDArray<double, 1> delta_weights, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;
        intr_x = weights.getDimensions()[0];
        intr_y = weights.getDimensions()[1];

        block.x = (intr_x > 32) ? 32 : intr_x;
        block.y = (intr_y > 32) ? 32 : intr_y;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixUpdateParameters<<<grid, block, 0, stream>>>(weights.getData(), learning_rate.getData(), delta_weights.getData(), intr_x, intr_y, 1);
    }

    void NDMath::updateBiases(NDArray<double, 1> biases, NDArray<double, 1> learning_rate, NDArray<double, 1> delta_biases, cudaStream_t stream)
    {
        unsigned intr_x;
        dim3 grid, block;
        intr_x = biases.getDimensions()[0];

        block.x = (intr_x > 32) ? 32 : intr_x;

        grid.x = ceil((float)intr_x / block.x);

        gpu::matrixUpdateParameters<<<grid, block, 0, stream>>>(biases.getData(), learning_rate.getData(), delta_biases.getData(), intr_x, 1, 1);
    }

    void NDMath::updateWeightsSGDmomentum(NDArray<double, 1> sigma, NDArray<double, 1> weights, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_weights, NDArray<double, 1> delta_weights, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;
        intr_x = weights.getDimensions()[0];
        intr_y = weights.getDimensions()[1];

        block.x = (intr_x > 32) ? 32 : intr_x;
        block.y = (intr_y > 32) ? 32 : intr_y;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixExponentiallyWeightedMovingAvg<<<grid, block, 0, stream>>>(*sigma.getData(), sum_delta_weights.getData(), delta_weights.getData(), intr_x, intr_y, 1);
    }

    void NDMath::updateBiasesSGDmomentum(NDArray<double, 1> sigma, NDArray<double, 1> biases, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_biases, NDArray<double, 1> delta_biases, cudaStream_t stream)
    {
        unsigned intr_x;
        dim3 grid, block;
        intr_x = biases.getDimensions()[0];

        block.x = (intr_x > 32) ? 32 : intr_x;

        grid.x = ceil((float)intr_x / block.x);

        gpu::matrixExponentiallyWeightedMovingAvg<<<grid, block, 0, stream>>>(*sigma.getData(), sum_delta_biases.getData(), delta_biases.getData(), intr_x, 1, 1);
    }

    void NDMath::updateWeightsRMSpropDense(NDArray<double, 1> sigma, NDArray<double, 1> epsalon, NDArray<double, 1> weights, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_weights, NDArray<double, 1> delta_weights, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;
        intr_x = weights.getDimensions()[0];
        intr_y = weights.getDimensions()[1];

        block.x = (intr_x > 32) ? 32 : intr_x;
        block.y = (intr_y > 32) ? 32 : intr_y;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixUpdateWeightsBiasesRMSprop<<<grid, block, 0, stream>>>(*sigma.getData(), *epsalon.getData(), sum_delta_weights.getData(), delta_weights.getData(), intr_x, intr_y, 1);
    }

    void NDMath::updateBiasesRMSpropDense(NDArray<double, 1> sigma, NDArray<double, 1> epsalon, NDArray<double, 1> biases, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_biases, NDArray<double, 1> delta_biases, cudaStream_t stream)
    {
        unsigned intr_x;
        dim3 grid, block;
        intr_x = biases.getDimensions()[0];

        block.x = (intr_x > 32) ? 32 : intr_x;

        grid.x = ceil((float)intr_x / block.x);

        gpu::matrixUpdateWeightsBiasesRMSprop<<<grid, block, 0, stream>>>(*sigma.getData(), *epsalon.getData(), sum_delta_biases.getData(), delta_biases.getData(), intr_x, 1, 1);
    }

    void NDMath::updateWeightsADAMDense(NDArray<double, 1> sigma, NDArray<double, 1> epsalon, NDArray<double, 1> weights, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_weights, NDArray<double, 1> sum_delta_weights_square, NDArray<double, 1> delta_weights, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;
        intr_x = weights.getDimensions()[0];
        intr_y = weights.getDimensions()[1];

        block.x = (intr_x > 32) ? 32 : intr_x;
        block.y = (intr_y > 32) ? 32 : intr_y;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixUpdateWeightsBiasesADAM<<<grid, block, 0, stream>>>(sigma.getData(), epsalon.getData(), learning_rate.getData(), sum_delta_weights.getData(), sum_delta_weights_square.getData(), delta_weights.getData(), weights.getData(), intr_x, intr_y);
    }

    void NDMath::updateBiasesADAMDense(NDArray<double, 1> sigma, NDArray<double, 1> epsalon, NDArray<double, 1> biases, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_biases, NDArray<double, 1> sum_delta_biases_squared, NDArray<double, 1> delta_biases, cudaStream_t stream)
    {
        unsigned intr_x;
        dim3 grid, block;
        intr_x = biases.getDimensions()[0];

        block.x = (intr_x > 32) ? 32 : intr_x;

        grid.x = ceil((float)intr_x / block.x);

        gpu::matrixUpdateWeightsBiasesADAM<<<grid, block, 0, stream>>>(sigma.getData(), epsalon.getData(), learning_rate.getData(), sum_delta_biases.getData(), sum_delta_biases_squared.getData(), delta_biases.getData(), biases.getData(), intr_x, 1);
    }

    void NDMath::getDifferentialWeights(NDArray<double, 1> input, NDArray<double, 1> delta_output, NDArray<double, 1> difference, NDArray<double, 1> delta_weights, NDArray<double, 1> delta_weights_intermediate, cudaStream_t stream)
    {
        int intr_x, intr_y, intr_z;
        dim3 grid, block;

        intr_x = difference.getDimensions()[0]; // no of output neuron
        intr_y = input.getDimensions()[0];      // no of input feature
        intr_z = input.getDimensions()[1];      // no of batches

        block.x = (intr_x > 8) ? 8 : intr_x;
        block.y = (intr_y > 8) ? 8 : intr_y;
        block.z = (intr_z > 16) ? 16 : intr_z;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);
        grid.z = ceil((float)intr_z / block.z);

        gpu::matrixDifferentialParameters<<<grid, block, 0, stream>>>(input.getData(), delta_output.getData(), difference.getData(), delta_weights_intermediate.getData(), intr_x, intr_y, intr_z);

        block.z = 1;
        grid.z = 1;

        gpu::matrixRollingSum<<<grid, block, 0, stream>>>(delta_weights_intermediate.getData(), delta_weights.getData(), intr_x, intr_y, intr_z);

        gpu::matrixFindMean<<<grid, block, 0, stream>>>(delta_weights.getData(), intr_x, intr_y, intr_z);
    }

    void NDMath::getDifferentialBiases(NDArray<double, 1> delta_output, NDArray<double, 1> difference, NDArray<double, 1> delta_biases, NDArray<double, 1> delta_biases_intermediate, cudaStream_t stream)
    {
        int intr_x, intr_y;
        dim3 grid, block;

        intr_x = delta_output.getDimensions()[0]; // no of output neuron
        intr_y = delta_output.getDimensions()[1]; // no of batches

        block.x = (intr_x > 8) ? 8 : intr_x;
        block.y = (intr_y > 8) ? 8 : intr_y;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixDifferentialBiases<<<grid, block, 0, stream>>>(delta_output.getData(), difference.getData(), delta_biases_intermediate.getData(), intr_x, intr_y);

        block.y = 1;
        grid.y = 1;

        gpu::matrixRollingSum<<<grid, block, 0, stream>>>(delta_biases_intermediate.getData(), delta_biases.getData(), intr_x, 1, intr_y);

        gpu::matrixFindMean<<<grid, block, 0, stream>>>(delta_biases.getData(), intr_x, 1, intr_y);
    }

    void NDMath::getDifferentialInput(NDArray<double, 1> weights, NDArray<double, 1> delta_output, NDArray<double, 1> difference, NDArray<double, 1> difference_input, NDArray<double, 1> delta_input_intermediate, NDArray<double, 1> delta_input, cudaStream_t stream)
    {
        // NDArray<double, 1> delta_input_intermediate;
        unsigned intr_x, intr_y, intr_z; // intr_x: no of input + bias, intr_y: batch size, intr_z = no of neuron
        dim3 grid, block;
        intr_x = delta_input.getDimensions()[0]; // no of input feature
        intr_y = delta_input.getDimensions()[1]; // no of batchs
        intr_z = weights.getDimensions()[0];     // no of neuron

        block.x = (intr_x > 8) ? 8 : intr_x;
        block.y = (intr_y > 8) ? 8 : intr_y;
        block.z = (intr_y > 16) ? 16 : intr_z;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);
        grid.z = ceil((float)intr_z / block.z);

        gpu::matrixDifferentialInput<<<grid, block, 0, stream>>>(weights.getData(), delta_output.getData(), difference.getData(), delta_input_intermediate.getData(), intr_x, intr_y, intr_z);

        block.z = 1;
        grid.z = 1;

        gpu::matrixRollingSum<<<grid, block, 0, stream>>>(delta_input_intermediate.getData(), difference_input.getData(), intr_x, intr_y, intr_z);
    }

    void NDMath::reluActivation(NDArray<double, 1> input, NDArray<double, 1> d_activation, cudaStream_t stream)
    {
        int intr_x, intr_y;
        dim3 block, grid;

        intr_x = input.getDimensions()[0];
        intr_y = input.getDimensions()[1];

        block.x = (intr_x > 32) ? 32 : intr_x;
        block.y = (intr_y > 32) ? 32 : intr_y;
        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixRelu<<<grid, block, 0, stream>>>(input.getData(), d_activation.getData(), intr_x, intr_y);
    }

    void NDMath::reluActivation(NDArray<double, 0> input)
    {
        // unsigned intr_x;

        // intr_x = input.getDimensions()[0];

        // cpu::matrixRelu(input.getData(), intr_x);
    }

    void NDMath::sigmoidActivation(NDArray<double, 1> input, NDArray<double, 1> d_activation, cudaStream_t stream)
    {
        int intr_x, intr_y;
        dim3 block, grid;
        intr_x = input.getDimensions()[0];
        intr_y = input.getDimensions()[1];

        block.x = ((intr_x + 1) > 32) ? 32 : (intr_x + 1);
        block.y = ((intr_y + 1) > 32) ? 32 : (intr_y + 1);
        grid.x = ceil((float)(intr_x + 1) / block.x);
        grid.y = ceil((float)(intr_y + 1) / block.y);

        gpu::matrixSigmoid<<<grid, block, 0, stream>>>(input.getData(), d_activation.getData(), intr_x, intr_y);
    }

    void NDMath::sigmoidActivation(NDArray<double, 0> input)
    {
        // unsigned intr_x;
        // intr_x = input.getDimensions()[0];
        // cpu::matrixSigmoid(input.getData(), intr_x);
    }

    void NDMath::linearActivation(NDArray<double, 1> input, NDArray<double, 1> d_activation, cudaStream_t stream)
    {
        int intr_x, intr_y;
        dim3 grid, block;

        intr_x = input.getDimensions()[0];
        intr_y = input.getDimensions()[1];

        block.x = ((intr_x + 1) > 32) ? 32 : (intr_x + 1);
        block.y = ((intr_y + 1) > 32) ? 32 : (intr_y + 1);
        grid.x = ceil((float)(intr_x + 1) / block.x);
        grid.y = ceil((float)(intr_y + 1) / block.y);

        gpu::matrixLinear<<<grid, block, 0, stream>>>(input.getData(), d_activation.getData(), intr_x, intr_y);
    }

    void NDMath::softmaxActivation(NDArray<double, 1> input, NDArray<double, 1> softmax_sum, NDArray<double, 1> d_activation, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;

        intr_x = input.getDimensions()[0];
        intr_y = input.getDimensions()[1];

        block.x = ((intr_x + 1) > 32) ? 32 : (intr_x + 1);
        block.y = ((intr_y + 1) > 32) ? 32 : (intr_y + 1);
        grid.x = ceil((float)(intr_x + 1) / block.x);
        grid.y = ceil((float)(intr_y + 1) / block.y);

        gpu::matrixSoftmax<<<grid, block, 0, stream>>>(input.getData(), softmax_sum.getData(), d_activation.getData(), intr_x, intr_y);
    }

    void NDMath::squaredError(NDArray<double, 1> Difference, NDArray<double, 1> Squared_Error, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;

        intr_x = Difference.getDimensions()[0];
        intr_y = Difference.getDimensions()[1];

        block.x = intr_x > 32 ? 32 : intr_x;
        block.y = intr_y > 32 ? 32 : intr_y;
        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixSquaredError<<<grid, block, 0, stream>>>(Difference.getData(), Squared_Error.getData(), intr_x, intr_y);
    }

    void NDMath::findMean(NDArray<double, 1> X, NDArray<double, 1> Y, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;
        intr_x = X.getDimensions()[0];
        intr_y = X.getDimensions()[1];

        gpu::matrixRollingSum<<<grid, block, 0, stream>>>(X.getData(), X.getData(), intr_x, 1, intr_y);
        Y.initData(X);

        gpu::matrixFindMean<<<grid, block, 0, stream>>>(Y.getData(), intr_x, intr_y, intr_y);
    }

    double NDMath::findMean(NDArray<double, 0> A)
    {
        double mean = 0.0;
        // cpu::getMean(A.getData(), &mean, A.getNoOfElem());
        return mean;
    }

    void NDMath::argMax(NDArray<double, 1> probabilities, NDArray<double, 1> one_hot_code, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;

        intr_x = probabilities.getDimensions()[0];
        intr_y = probabilities.getDimensions()[1];

        block.x = 1;
        block.y = intr_y > 32 ? 32 : intr_y;

        grid.x = 1;
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixArgMax<<<grid, block, 0, stream>>>(probabilities.getData(), one_hot_code.getData(), intr_x, intr_y);
    }

    NDArray<double, 0> NDMath::findSquare(NDArray<double, 0> A)
    {
        NDArray<double, 0> result(A.getNoOfDimensions(), A.getDimensions());
        // cpu::getSquare(A.getData(), result.getData(), A.getNoOfElem());
        return result;
    }

    NDArray<double, 0> NDMath::findSquareRoot(NDArray<double, 0> A)
    {
        NDArray<double, 0> result(A.getNoOfDimensions(), A.getDimensions());
        // cpu::getSquareRoot(A.getData(), result.getData(), A.getNoOfElem());
        return result;
    }

    void NDMath::findDifference(NDArray<double, 1> Y_predict, NDArray<double, 1> Y_target, NDArray<double, 1> Difference, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        double scaler_value = 2.0;
        dim3 grid, block;

        intr_x = Y_target.getDimensions()[0];
        intr_y = Y_target.getDimensions()[1];

        block.x = intr_x > 32 ? 32 : intr_x;
        block.y = intr_y > 32 ? 32 : intr_y;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixDifference<<<grid, block, 0, stream>>>(Y_predict.getData(), Y_target.getData(), Difference.getData(), intr_x, intr_y);
        gpu::matrixScalerMul<<<grid, block, 0, stream>>>(Difference.getData(), scaler_value, Difference.getData(),intr_x, intr_y, 1);
    }

    NDArray<double, 0> NDMath::findDifference(NDArray<double, 0> A, NDArray<double, 0> B)
    {
        NDArray<double, 0> result;
        unsigned nDim, flag;

        nDim = A.getNoOfDimensions();
        flag = 1;

        if (nDim == B.getNoOfDimensions())
        {
            for (int i = 0; i < nDim; i++)
            {
                if (A.getDimensions()[i] != B.getDimensions()[i])
                {
                    flag = 0;
                    break;
                }
            }
            if (flag)
            {
                result = NDArray<double, 0>(nDim, A.getDimensions());
                // cpu::getDifference(A.getData(), B.getData(), result.getData(), A.getNoOfElem());
            }
        }

        return result;
    }

    // void NDMath::crossEntropy(NDArray<double, 1> Y_predict, NDArray<double, 1> Y_target, NDArray<double, 1> Difference, NDArray<double, 1> Cost, cudaStream_t stream)
    // {
    //     unsigned intr_x, intr_y;
    //     dim3 grid, block;

    //     intr_x = Y_target.getDimensions()[0];
    //     intr_y = Y_target.getDimensions()[1];

    //     block.x = intr_x > 32 ? 32 : intr_x;
    //     block.y = intr_y > 32 ? 32 : intr_y;

    //     grid.x = ceil((float)intr_x / block.x);
    //     grid.y = ceil((float)intr_y / block.y);

    //     gpu::matrixCrossEntropyDifference<<<grid, block, 0, stream>>>(Y_predict.getData(), Y_target.getData(), Difference.getData(), intr_x, intr_y);

    //     block.x = 1;
    //     grid.x = 1;

    //     gpu::matrixCrossEntropy<<<grid, block, intr_y * sizeof(double), stream>>>(Y_predict.getData(), Y_target.getData(), Cost.getData(), intr_x, intr_y);
    // }

    void NDMath::binaryCrossEntropy(NDArray<double, 1> Y_predict, NDArray<double, 1> Y_target, NDArray<double, 1> Difference, NDArray<double, 1> Cost, cudaStream_t stream)
    {
        unsigned intr_x, intr_y, intr_x_predict;
        dim3 grid, block;

        intr_x = Y_target.getDimensions()[0];
        intr_y = Y_target.getDimensions()[1];
        intr_x_predict = Y_predict.getDimensions()[0];

        if (intr_x == 1 && intr_x_predict == 1)
        {
            block.x = intr_x > 32 ? 32 : intr_x;
            block.y = intr_y > 32 ? 32 : intr_y;

            grid.x = ceil((float)intr_x / block.x);
            grid.y = ceil((float)intr_y / block.y);

            gpu::matrixBinaryCrossEntropy<<<grid, block, 0, stream>>>(Y_predict.getData(), Y_target.getData(), Difference.getData(), intr_x, intr_y);
        }
    }

    void NDMath::confusionMatrix(NDArray<double, 1> predict, NDArray<double, 1> actual, NDArray<double, 1> confusion_matrix, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;

        intr_x = predict.getDimensions()[0];
        intr_y = predict.getDimensions()[1];

        block.x = intr_x > 32 ? 32 : intr_x;
        block.y = intr_x > 32 ? 32 : intr_x;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_x / block.x);

        gpu::matrixConfusionMatrix<<<grid, block, 0, stream>>>(predict.getData(), actual.getData(), confusion_matrix.getData(), intr_x, intr_y);
    }

    void NDMath::accuracyValue(NDArray<double, 1> confusion_matrix, NDArray<double, 1> accuracy_value, unsigned no_of_classes, unsigned no_of_samples, cudaStream_t stream)
    {
        gpu::matrixAccuracyValue<<<1, 1, 0, stream>>>(confusion_matrix.getData(), accuracy_value.getData(), no_of_classes, no_of_samples);
    }

