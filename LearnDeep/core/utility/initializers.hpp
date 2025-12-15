#ifndef _TENSORFLOW_CORE_UTILITY_INITILIZERS_
#define _TENSORFLOW_CORE_UTILITY_INITILIZERS_

#include <random>

namespace util {

class random_engine {
  std::random_device rd;

  std::mt19937 gen;

public:
  random_engine() : gen(rd()) {}
  void generate_uniform_random(double *ptr, size_t nElem, double upper_bound,
                               double lower_bound);

  void generate_normal_random(double *ptr, size_t nElem, double upper_bound,
                              double lower_bound);

  int rand_int(int lower_bound, int upper_bound);

  unsigned rand_unsigned(unsigned lower_bound, unsigned upper_bound);
};

} // namespace util

#endif // _TENSORFLOW_CORE_UTILITY_INITILIZERS_