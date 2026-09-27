A Catalogue of Optimizing Transformations

A systematic catalogue of program transformations that preserve the meaning of
a program while improving its performance. This paper predates modern
compilers and is largely of historical interest.

The catalogue is organised by the kind of transformation: common subexpression
elimination, loop invariant code motion, strength reduction, induction
variable elimination, and dead code elimination.

The paper's central claim is that optimizing transformations compose badly.
Two individually beneficial transformations can interact to produce a program
that is slower than either input, which is why whole-program analysis became
the dominant approach in later compilers.

Modern compilers use a cost model to decide whether a transformation is worth
applying, since an optimisation that always fires is not automatically an
optimisation.
