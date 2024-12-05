#pragma ONCE

namespace cpu
{

    void __mmul(double **, unsigned *);

    void __melementwisemul(double **, unsigned *);

    void __mmulconventional(double **, unsigned *);

    void __mscalermul(double **, unsigned *);

    void __madd(double **, unsigned *);

    void __msub(double **, unsigned *);

    void __mrollingsum(double **, unsigned *);

    void __mtranspose(double **, unsigned*);
}
