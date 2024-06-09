#pragma ONCE

typedef struct struct_Activations
{
    enum activationFunc
    {
        relu,
        sigmoid,
        softmax,
        tanh,
        e_relu,
        leaky_relu,
        linear
    };
    std::map<std::string, activationFunc> activations;

    struct_Activations()
    {
        activations["relu"] = relu;
        activations["sigmoid"] = sigmoid;
        activations["softmax"] = softmax;
        activations["e_relu"] = e_relu;
        activations["leaky_relu"] = leaky_relu;
        activations["tanh"] = tanh;
        activations["linear"] = linear;
    }
} struct_Activations;

class Activation
{
protected:
    NDMath math;

public:
    virtual void activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream){} 
    virtual void activate(NDArray<double, 0> output) {}
};

class Relu_Activation : public Activation
{
public:
    void activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream) override;
    void activate(NDArray<double, 0> output) override;
};

class Sigmoid_Activation : public Activation
{
public:
    void activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream) override;
    void activate(NDArray<double, 0> output) override;
};

class Linear_Activation : public Activation
{
public:
    void activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream) override;
    void activate(NDArray<double, 0> output) override;
};

class Softmax_Activation : public Activation
{
    NDArray<double, 1> softmax_sum;
    unsigned flag = 0;

public:
    void activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream) override;

    void activate(NDArray<double, 0> output) override;
};
