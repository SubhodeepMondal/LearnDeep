#include "kernelmanager.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

KernelType parse_kernel_env() {
  const char *env = std::getenv("DL_KERNEL");
  if (!env)
    return KernelType::AUTO;

  std::string val(env);
  std::transform(val.begin(), val.end(), val.begin(), ::toupper);

  if (val == "CPU" || val == "SCALAR")
    return KernelType::CPU_SCALAR;
  if (val == "AVX2")
    return KernelType::AVX2;
  if (val == "GPU" || val == "CUDA")
    return KernelType::GPU;

  return KernelType::AUTO; // fallback safety
}

KernelType get_global_kernel() {
  static KernelType kernel = parse_kernel_env();
  log_kernel_choice();
  return kernel;
}

void log_kernel_choice() {
  static bool logged = false;
  if (!logged) {
    logged = true;
    std::cout << "[Kernel] Using: ";

    switch (get_global_kernel()) {
    case KernelType::CPU_SCALAR:
      std::cout << "CPU_SCALAR\n";
      break;
    case KernelType::AVX2:
      std::cout << "CPU_AVX2\n";
      break;
    case KernelType::GPU:
      std::cout << "GPU\n";
      break;
    default:
      std::cout << "AUTO\n";
      break;
    }
  }
}