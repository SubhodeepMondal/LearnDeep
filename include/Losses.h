#pragma ONCE

typedef struct struct_Loss
{
    enum losses
    {
        categorical_crossentropy,
        binary_crossentropy,
        mean_squared_error,
        mean_abs_error,
        linear
    };
    std::map<std::string, losses> loss;

    struct_Loss()
    {
        loss["categorical_crossentropy"] = categorical_crossentropy;
        loss["binary_crossentropy"] = binary_crossentropy;
        loss["mean_squared_error"] = mean_squared_error;
        loss["mean_abs_error"] = mean_abs_error;
        loss["linear"] = linear;
    }
} struct_Loss;

class Loss
{
protected:
    unsigned *axis_dims;
    NDMath math;

public:
    virtual void findLoss(NDArray<double, 1> Y_predict, NDArray<double, 1> Y_target, NDArray<double, 1> Difference, NDArray<double, 1> Cost, cudaStream_t stream) {}
};


class Linear : public Loss
{
public:
    void findLoss(NDArray<double, 1> Y_predict, NDArray<double, 1> Y_target, NDArray<double, 1> Difference, NDArray<double, 1> Cost, cudaStream_t stream) override;
};


class Mean_Squared_Error : public Loss
{
    unsigned isSquaredErrorAlocated = 0;
    NDArray<double, 1> squared_error;

    void meanError(NDArray<double, 1> Difference, NDArray<double, 1> Cost, cudaStream_t stream);

public:
    void findLoss(NDArray<double, 1> Y_predict, NDArray<double, 1> Y_target, NDArray<double, 1> Difference, NDArray<double, 1> Cost, cudaStream_t stream) override;
};


class Catagorical_Cross_Entropy : public Loss
{
    unsigned isCrossEntropyErrorAlocated = 0;
    NDArray<double, 1> cross_entropy_error;

    void crossEntropyError(NDArray<double, 1>Y_predict, NDArray<double, 1>Y_target,NDArray<double, 1> Cost,NDArray<double, 1>Difference, cudaStream_t stream );

    public:
        void findLoss(NDArray<double, 1>Y_predict, NDArray<double, 1>Y_target, NDArray<double, 1>Difference, NDArray<double, 1> Cost, cudaStream_t stream) override;
};


class Binary_Cross_Entropy : public Loss
{
    public:
        void findLoss(NDArray<double, 1>Y_predict, NDArray<double, 1>Y_target, NDArray<double, 1>Difference, NDArray<double, 1>Cost, cudaStream_t stream) override;
};