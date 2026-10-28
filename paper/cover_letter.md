# Cover letter (draft)

To the Editors of *Computer Physics Communications*,

We submit for consideration as a Computer Programs in Physics (CPiP) article
the manuscript
**"Certifying neural eigenpairs and degenerate eigenspaces of the Schr\"odinger
equation: a self-verifying a posteriori protocol and its implementation"**
by Chunyang Chen and Yuhao Peng (equal contribution), School of Mathematics
and Physics, China University of Geosciences (Wuhan).

Neural-network solvers of the stationary Schr\"odinger equation are normally
evaluated by comparing with a reference solution. In a genuinely new problem
that reference does not exist, and a small residual does not bound the error
of the eigenfunction: in our own runs a residual of $4\times10^{-12}$ belonged
to one arbitrary member of a degenerate pair, and independent runs disagreed
about which member they had found (pairwise overlaps of $0.01$). The manuscript turns
this evaluation into a computation. It presents a protocol, and a public
implementation, that takes a network output on a finite-difference grid and
returns (i) an interval *verified* to contain exactly one eigenvalue of the
discrete operator, (ii) a two-sided enclosure of that eigenvalue, (iii) a bound
on the angle to the exact eigenfunction, and (iv) a verified state index. The
hypotheses of the classical Kato--Temple and Davis--Kahan theorems are decided
by counting the spectrum with Sturm sequences (1D) and with the Sylvester
inertia of a no-pivot sparse LU factorization (2D), with guards against silent
pivoting and a resolution floor below which the interval is refused.

The part we believe is new is not an inequality but a construction: because a
degenerate level defines no single eigenfunction, the certificate is extended
from states to invariant subspaces. The multiplicity is *measured* rather than
assumed, the cluster receives a two-sided enclosure at the counting
resolution, and a principal-angle bound certifies the subspace that a set of
network outputs spans, without any hypothesis on their mutual separation.
Using it we can decide, for four independently trained networks on the same
degenerate level, which pairs actually span that level: three had collapsed
onto the same symmetry-broken representative. The certificate refuses their
pair, accepts every pair that contains the orthogonal partner with bounds of
$5.2$--$6.3\times10^{-3}$, and prices the ambiguous pair---one that omits the
partner but whose second direction is mostly noise---at $0.68$, so that the
weakness is visible in the number rather than hidden. A
controlled splitting of the level shows that the boundary between individual
and subspace certification sits at the counting resolution, not at
degeneracy: in our scan the state certificate refuses at 0.95 times the
resolution and accepts at 1.01 times it. Guaranteed cluster bounds for
conforming discretizations exist (Cancès et al., Math. Comp. 2020); the
construction here does not assume Galerkin structure, which is what allows it
to apply to a network output.

Two further constructions close the caveats a referee would otherwise raise.
First, the count itself is re-derived in directed-rounding interval arithmetic:
the one-dimensional Sturm recurrence, and the two-dimensional count whenever
the potential separates as $V_x+V_y$ (where the operator is a Kronecker sum
and the count is a comparison of eigenvalue sums), are evaluated with every
rounding forced outward, so that a certified count is a theorem about the
stored matrix and an unresolved pivot is refused by index rather than rounded
to an integer; bisecting on the certified count yields certified brackets for
the discrete eigenvalues. The general two-dimensional interval elimination is
measured, reported and *not* used, because it refuses on most shifts. Second,
because every statement above concerns the discretization, the manuscript adds
a bound on the continuous operator: the interpolant of a sampled
eigenfunction---a network's included---is a conforming trial function, so its
Rayleigh quotient bounds the continuous eigenvalue from above with no
reference solution and no interpolation constant. The matching lower bound
needs a comparison principle; we give it for the infinite well and the box,
and we show it *fails* for the finite well, where a certified discrete energy
can lie above the continuous one because the stencil is not variationally
consistent at a jump.

The audit is backed by references computed without the certificate: 39 of 39
accepted enclosures contain a discrete eigenvalue obtained independently---a
dense LAPACK solve in two dimensions, never the ARPACK values that propose the
intervals---and the counting is cross-checked against dense diagonalization in
both dimensions.

The submission includes a 143-test suite, the drivers that regenerate every
table and figure of the article, and an appendix recording three reproducible
training failures---each with its saved run---that make a network look fine
while it cannot be certified. The package is released under the MIT licence,
which is on the journal's list of approved licences, and its Program Summary
follows the CPiP template; the source archive, a README describing the
directory structure, and a test run are uploaded with the manuscript. We state the limitations explicitly: every
two-sided enclosure refers to the discretized operator, and the bridge to the
continuous operator is one-sided; the interval count covers one dimension and
separable two-dimensional potentials, while the general two-dimensional count
remains floating point with guards; and the protocol does not apply when the
discrete operator cannot be assembled.

The manuscript is not under consideration elsewhere and has not been
published. We do not suggest referees. The authors declare no competing
interests. The code, the drivers that regenerate every table and figure, and
the 143-test suite are available at
https://github.com/YuhaoPeng1103/PINN_equation (MIT licence).

Yours sincerely,

Yuhao Peng (corresponding author), on behalf of both authors\\
School of Mathematics and Physics, China University of Geosciences (Wuhan)\\
pngyuo@cug.edu.cn
