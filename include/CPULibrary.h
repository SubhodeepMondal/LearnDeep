#pragma ONCE

namespace cpu
{
    void initilizeData(double *data, int nElem)
    {
        time_t t;
        srand((unsigned)time(&t));

        for (int i = 0; i < nElem; i++)
            data[i] = (float)(rand() & 0xFF) / 117.00000;
    }

    void matrixMul(double *A, double *B, double *C, int a_m, int a_n, int b_m, int b_n)
    {
        double sum = 0;
        for (int i = 0; i < a_m; i++)
            for (int j = 0; j < b_n; j++)
            {
                sum = 0;
                for (int k = 0; k < a_n; k++)
                    sum += A[i * a_n + k] * B[k * b_n + j];
                C[i * b_n + j] = sum;
            }
    }

    void compareArray(double *A, double *B, int n)
    {
        int flag = 0;
        for (int i = 0; i < n; i++)
            if (abs(A[i] - B[i]) > 0.01f)
            {
                flag = 1;
                break;
            }
        if (flag == 0)
            std::cout << "The arrays are exact match." << std::endl;
        else
            std::cout << "The arrays are not a match." << std::endl;
    }

    void matrixDotMul(double *A, double *B, double *C, double *D, unsigned a_m, unsigned a_n)
    {
        unsigned i, j, indx_W; // indx_A: index input A, indx_B: index weights B
        double val;

        for (i = 0; i < a_m; i++)
        {
            val = 0;
            for (j = 0; j < a_n; j++)
            {
                indx_W = i + j * a_m;
                val += A[j] * B[indx_W];
            }
            val += C[i];
            D[i] = val;
            // std::cout << D[i] << ", ";
        }
        // std::cout << "\n";
    }

    void matrixRelu(double *A, unsigned a_m)
    {
        unsigned i;
        for (i = 0; i < a_m; i++)
            if (A[i] < 0)
                A[i] = 0;
    }

    void matrixSigmoid(double *A, unsigned a_m)
    {
        unsigned i;
        for (i = 0; i < a_m; i++)
            A[i] = 1.0f / (1 + exp(-1 * A[i]));
    }

    void getMean(double *A, double *B, unsigned nElem)
    {
        double sum = 0.0;
        for (int i = 0; i < nElem; i++)
            sum += A[i];

        *B = sum / nElem;
    }

    void getDifference(double *A, double *B, double *C, unsigned nElem)
    {
        for (unsigned i = 0; i < nElem; i++)
            C[i] = A[i] - B[i];
    }

    void getSquare(double *A, double *B, unsigned nElem)
    {
        for (unsigned i = 0; i < nElem; i++)
            B[i] = A[i] * A[i];
    }

    void getSquareRoot(double *A, double *B, unsigned nElem)
    {
        for (unsigned i = 0; i < nElem; i++)
            B[i] = sqrt(A[i]);
    }

}
