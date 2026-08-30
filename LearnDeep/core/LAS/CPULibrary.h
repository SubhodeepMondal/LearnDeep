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

/**
 * @brief does reduction sum on a given axis
 * @param ptr pointer to pointer of long float pointing to
 *            ptr[0] input
 *            ptr[1] output
 * @param arr enoded integers
 *            arr[0]: n no of dimensions
 *            arr[1-n]: value of each dimensions
 *            arr[n]: reduction axis
 * @return void this funciton returns nothing
 */
void __mreducesum(std::float64_t *const *const, unsigned const *const);

void __mrelu(std::float64_t **ptr, unsigned const nDim, unsigned const *arr);

void __msigmoid(std::float64_t **, unsigned *);

void __msoftmax(std::float64_t *const *const, unsigned const axis,
                unsigned const vector_length, unsigned const inner_stride,
                unsigned const outer_stride);

void __mgreaterthanzero(std::float64_t *const *const ptr, unsigned const *dims,
                        unsigned const nDims);
} // namespace cpu
#endif // CPULIBRARY_H
