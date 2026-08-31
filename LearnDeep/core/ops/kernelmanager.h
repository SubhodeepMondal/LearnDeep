#ifndef KERNEL_MANAGER
#define KERNEL_MANAGER

enum class KernelType { AUTO, AVX, AVX2, AVX512, CPU_SCALAR, GPU };

KernelType parse_kernel_env();

KernelType get_global_kernel();

void log_kernel_choice();

#endif // KERNEL_MANAGER