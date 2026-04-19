#ifndef CPULIBRARY_H
#define CPULIBRARY_H

// C++ Headers
#include <cstddef>
#include <stdfloat>

namespace cpu {

void __matmul(std::float64_t **, unsigned *);

void __melementwisemul(std::float64_t **, const unsigned *);

void __mmul_broadcast(std::float64_t *const *const ptr, const unsigned nDimA,
                      const unsigned *dimA, const unsigned nDimB,
                      const unsigned *dimB, const bool isBroadCast);

void __matmul_conventional(std::float64_t **, unsigned *);

void __mscalermul(std::float64_t **, unsigned *);

void __mscalermul_broadcast(std::float64_t *const *const ptr,
                            const unsigned nDimA, const unsigned *dimA,
                            const unsigned nDimB, const unsigned *dimB,
                            const bool isBroadCast);

void __madd(std::float64_t **, unsigned *);

void __madd_broadcast(std::float64_t *const *const ptr, const unsigned nDimA,
                      const unsigned *dimA, const unsigned nDimB,
                      const unsigned *dimB, const bool isBroadCast);

void __msub(std::float64_t **, unsigned *);

void __msub_broadcast(std::float64_t *const *const ptr, const unsigned nDimA,
                      const unsigned *dimA, const unsigned nDimB,
                      const unsigned *dimB, const bool isBroadCast);

void __mrollingsum(std::float64_t **, unsigned *);

void __mtranspose(std::float64_t **, unsigned *);

void __mtiled_transpose(std::float64_t **, unsigned *);

void __msqrt(std::float64_t **, unsigned *);

void __mrelu(std::float64_t **, unsigned *);

void __msigmoid(std::float64_t **, unsigned *);

void __msoftmax(std::float64_t **, unsigned *);
} // namespace cpu
#endif // CPULIBRARY_H