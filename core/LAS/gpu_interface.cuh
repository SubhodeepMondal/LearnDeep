#ifndef GPU_INTERFACE_CUH
#define GPU_INTERFACE_CUH
namespace gpu
{
    // Function to perform matrix addition
    // This function adds two matrices of type double and stores the result in a third matrix
    // The matrices are represented as pointers to double arrays, and their dimensions are specified by the
    // parameters in the 'arr' array, where arr[0] is the number of rows and arr[1] is the number of columns.
    // The function uses CUDA to perform the addition on the GPU.
    // It allocates memory on the GPU for the input matrices and the output matrix,
    // copies the input matrices from the host to the device, launches a CUDA kernel to perform the addition,
    // and finally copies the result back to the host before freeing the GPU memory.
    // The function also checks for CUDA errors and prints an error message if any occur.
    // Parameters:
    // - ptr: An array of pointers to double arrays, where ptr[0] is the first matrix,
    //        ptr[1] is the second matrix, and ptr[2    ] is the result matrix.
    // - arr: An array of unsigned integers where arr[0] is the number of rows and arr[1] is the number of columns of the matrices.
    // Returns: void
    // Note: The function assumes that the input matrices are of the same dimensions.
    // Note: The function uses a block size of 32x32 for the CUDA kernel
    //       and calculates the grid size based on the dimensions of the matrices.
    // Note: The function uses double precision floating point numbers (double).
    void gpu_mat_add_f64(double **ptr, unsigned *arr);


    // Function to perform element-wise multiplication of two matrices
    // This function multiplies two matrices of type double element-wise and stores the result in a third matrix.
    // The matrices are represented as pointers to double arrays, and their dimensions are specified by the
    // parameters in the 'arr' array, where arr[0] is the number of rows and arr[1] is the number of columns.
    // The function uses CUDA to perform the multiplication on the GPU.
    // It allocates memory on the GPU for the input matrices and the output matrix,
    // copies the input matrices from the host to the device, launches a CUDA kernel to perform the multiplication,
    // and finally copies the result back to the host before freeing the GPU memory.
    // The function also checks for CUDA errors and prints an error message if any occur.
    // Parameters:
    // - ptr: An array of pointers to double arrays, where ptr[0] is the first matrix,
    //        ptr[1] is the second matrix, and ptr[2] is the result matrix.
    // - arr: An array of unsigned integers where arr[0] is the number of rows and arr[1] is the number of columns of the matrices.
    void gpu_mat_hadamard_mul_f64(double **ptr, unsigned *arr);

    // Function to perform matrix multiplication
    // Note: The function name has been changed to avoid confusion with element-wise multiplication
    // This function multiplies two matrices of type double and stores the result in a third matrix.
    // The matrices are represented as pointers to double arrays, and their dimensions are specified by the
    // parameters in the 'arr' array, where arr[0] is the number of rows and arr[1] is the number of columns.
    // The function uses CUDA to perform the multiplication on the GPU.
    // It allocates memory on the GPU for the input matrices and the output matrix,
    // copies the input matrices from the host to the device, launches a CUDA kernel to perform the multiplication,
    // and finally copies the result back to the host before freeing the GPU memory.
    // The function also checks for CUDA errors and prints an error message if any occur        .
    // Parameters:
    // - ptr: An array of pointers to double arrays, where ptr[0] is the first matrix,
    //        ptr[1] is the second matrix, and ptr[2] is the result matrix.
    // - arr: An array of unsigned integers where arr[0] is the number of rows and arr[1] is the number of columns of the matrices. 
    void gpu_mat_mul_f64(double **ptr, unsigned *arr);
}
#endif // GPU_INTERFACE_CUH