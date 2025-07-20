#ifndef CPULIBRARY_H
#define CPULIBRARY_H

#include <stdfloat>

namespace cpu
{

    void __matmul(std::float64_t **, unsigned *);

    void __melementwisemul(std::float64_t **, unsigned *);

    void __matmul_conventional(std::float64_t **, unsigned *);

    void __mscalermul(std::float64_t **, unsigned *);

    void __madd(std::float64_t **, unsigned *);

    void __msub(std::float64_t **, unsigned *);

    void __mrollingsum(std::float64_t **, unsigned *);

    void __mtranspose(std::float64_t **, unsigned*);
}
#endif // CPULIBRARY_H