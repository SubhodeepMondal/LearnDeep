#ifndef GPU_INTERFACE_CUH
#define GPU_INTERFACE_CUH
namespace gpu
{
    void gpu_mat_add_f64(double **ptr, unsigned *arr);

    void gpu_mat_mul_f64(double **ptr, unsigned *arr);
}
#endif // GPU_INTERFACE_CUH