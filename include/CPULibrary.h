#pragma ONCE

namespace cpu
{

    void __mmul(double **, unsigned *);

    void __mmulconventional(double **, unsigned *);

    void __mscalermul(double *, double, double *, unsigned, unsigned);

    void __madd(double **, unsigned *);

    void __msub(double **, unsigned*);

    void __mrollingsum(double *, double *, unsigned, unsigned, unsigned, unsigned);

    void __mtranspose(double *, double *, unsigned, unsigned);
}
