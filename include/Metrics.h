#pragma ONCE

typedef struct struct_Metrics
{
    enum metrics
    {
        accuracy,
        precision,
        recall,
        f1_score

    };
    std::map<std::string, metrics> metric;

    struct_Metrics()
    {
        metric["accuracy"] = accuracy;
        metric["precision"] = precision;
        metric["recall"] = recall;
        metric["f1_score"] = f1_score;
    }
} struct_Metric;

class Metric
{
protected:
    NDArray<double, 1> confusion_matrix;
    unsigned flag = 0, no_of_samples, no_of_classes;

protected:
    NDMath math;
    void confusionMatrix(NDArray<double, 1> predict, NDArray<double, 1> actual, cudaStream_t stream);

public:
    virtual void accuracy(NDArray<double, 1> predict, NDArray<double, 1> actual, NDArray<double, 1> accuracy_value, cudaStream_t stream);
};

class Accuracy : public Metric
{
    void accuracy(NDArray<double, 1> predict, NDArray<double, 1> actual, NDArray<double, 1> accuracy_value, cudaStream_t stream) override;
};