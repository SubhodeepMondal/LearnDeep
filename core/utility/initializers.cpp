// Library Headers
#include "initializers.hpp"

void util::random_engine::generate_uniform_random(double *ptr, size_t nElem,
                                                  double lower_bound,
                                                  double upper_bound) {
  std::uniform_real_distribution<double> uniform_double(lower_bound,
                                                        upper_bound);
  for (int i = 0; i < nElem; i++) {
    ptr[i] = uniform_double(gen);
  }
}

void util::random_engine::generate_normal_random(double *ptr, size_t nElem,
                                                 double lower_bound,
                                                 double upper_bound) {
  std::normal_distribution<double> normal_double(lower_bound, upper_bound);

  for (int i = 0; i < nElem; i++)
    ptr[i] = normal_double(gen);
}

int util::random_engine::rand_int(int lower_bound, int upper_bound) {
  std::uniform_int_distribution<int> uniform_int(lower_bound, upper_bound);

  return uniform_int(gen);
}

unsigned util::random_engine::rand_unsigned(unsigned lower_bound,
                                            unsigned upper_bound) {
  std::uniform_int_distribution<unsigned> uniform_unsigned(lower_bound,
                                                           upper_bound);

  return uniform_unsigned(gen);
}