# Function-Projection

We find the best polynomial approximation of degree `n` for a continuous function defined on an interval [a,b]. It's a fun linear algebra thing.

Consider C[a,b], the set of continuous functions defined on [a,b], which is actually an inner product space. We can add functions, scale functions, and get an inner product by integrating along their product from a to b.

Then, we consider the subspace of C[a,b], the space of real polynomials of degree `n`. We come up with an orthogonal basis for this subspace, then to approximate a function f we project it onto this subspace.
