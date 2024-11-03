#include<map>
#include"NDynamicArray.h"
#include "MathLibrary.h"
#include "Metrics.h"

void Metric::confusionMatrix(NDArray<double, 1> predict, NDArray<double, 1> actual, cudaStream_t stream)
{
    math.confusionMatrix(predict, actual, confusion_matrix, stream);
}

void Metric::accuracy(NDArray<double, 1> predict, NDArray<double, 1> actual, NDArray<double, 1> accuracy_value, cudaStream_t stream) {};

void Accuracy::accuracy(NDArray<double, 1> predict, NDArray<double, 1> actual, NDArray<double, 1> accuracy_value, cudaStream_t stream)
{
    no_of_classes = predict.getDimensions()[0];
    no_of_samples = predict.getDimensions()[1];

    if (predict.getNoOfDimensions() == 2 && actual.getNoOfDimensions() == predict.getNoOfDimensions())
    {
        if (predict.getDimensions()[0] == actual.getDimensions()[0])
        {
            if (!flag)
            {
                no_of_classes = predict.getDimensions()[0];
                confusion_matrix = NDArray<double, 1>(2, no_of_classes, no_of_classes);
                flag = 1;
            }

            confusionMatrix(predict, actual, stream);
            math.accuracyValue(confusion_matrix, accuracy_value, no_of_classes, no_of_samples, stream);
        }
    }
}
