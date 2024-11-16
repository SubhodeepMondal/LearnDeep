#include <iostream>
#include "deeplearning.h"


void generatorFunctionCluster(NDArray<double, 0> x, NDArray<double, 0> y)
{
    double x1, y1;
    unsigned i, j, no_of_features, no_of_samples;
    double *ptrA, *ptrB;

    no_of_features = x.getDimensions()[0];
    no_of_samples = x.getDimensions()[1];

    ptrA = x.getData();
    ptrB = y.getData();

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 5.0);

    i = 0;

    std::cout << "Generating cluster of data: .............\n";

    while (i <= no_of_samples)
    {
        x1 = distribution(generator);
        y1 = distribution(generator);

        if (y1 < (1.5 * x1 - 3) && y1 > (-0.5 * x1 + 2))
        {
            j = 0;
            ptrA[i * no_of_features + j] =x1;
            j += 1;
            ptrA[i * no_of_features + j] = y1;

            j = 0;
            ptrB[i * no_of_features + j] = 1;
            j+= 1;
            ptrB[i * no_of_features + j] = 0;
            i++;
        }

        
        else if (y1 > (1.5 * x1 - 3) && y1 < (-0.5 * x1 + 2))
        {
            j = 0;
            ptrA[i * no_of_features + j] = x1;
            j+= 1;
            ptrA[i * no_of_features + j] = y1;

            j = 0;
            ptrB[i * no_of_features + j] = 0;
            j+= 1;
            ptrB[i * no_of_features + j] = 1;
            i++;
        }
    }
}

int main()
{
    NDMath math;
    freopen("io/input.txt", "r", stdin);
    freopen("io/DLearning.csv", "w", stdout);
    Metric *metric = new Accuracy;
    unsigned no_of_feature, no_of_sample, epochs, batch_size;
    NDArray<double, 1> y_predict_proba, y_predict, accuracy;
    no_of_sample = 1000;
    no_of_feature = 2;

    std::cin >> epochs;
    std::cin >> batch_size;

    NDArray<double, 0> X_train(2, no_of_feature, no_of_sample);
    NDArray<double, 0> y_train(2, 2, no_of_sample);


    NDArray<double, 0> X_test(2, no_of_feature, 100);
    NDArray<double, 0> y_test(2, 2, 100);



    // X_train.printData();
    // y_train.printData();

    // generatorFunctionCluster(X_train, y_train);
    // generatorFunctionCluster(X_test, y_test);

    // X_test.printData();
    // y_test.printData();

    NDArray<double, 0> input_shape(1, &no_of_feature);

    Model *model = new Sequential();
    model->add(new Dense{8, input_shape, "relu", "Dense_1"});
    model->add(new Dense{2, "softmax", "Dense_2"});
    model->compile("categorical_crossentropy", "sgd", "accuracy");
    model->summary();

    model->fit(X_train, y_train, epochs, batch_size);

    // y_predict_proba = NDArray<double, 1>(y_test.getNoOfDimensions(), y_test.getDimensions());
    // y_predict = NDArray<double, 1>(y_test.getNoOfDimensions(), y_test.getDimensions());
    // accuracy = NDArray<double, 1>(1,1);
    // y_predict_proba = model->predict(X_test);
    // math.argMax(y_predict_proba, y_predict, NULL);
    // metric->accuracy(y_predict, y_test, accuracy, NULL);
    // accuracy.printData();
}