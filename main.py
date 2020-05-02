import numpy as np
import math
import scipy.integrate as integrate


def main():
    n = 7
    a = -math.pi
    b = math.pi

    inner_prod = polynomial_inner_prod_matrix(n, a, b)


    print(standard_basis(n)[0])

    basis = gram_shmidt(
        standard_basis(n),
        inner_prod
    )

    projection = project(math.cos, a, b, basis, inner_prod)
    print(output_for_desmos(projection))


def standard_basis(n):
    basis = []
    for i in range(n):
        x = np.zeros(n)
        x[i] = 1.0
        basis.append(x)
    return basis


def project(f, a, b, basis, inner_prod):
    p = np.zeros(inner_prod.shape[0])
    for q in basis:
        p = p + function_polynomial_inner_prod(a, b, f, q) * q

    return p


def gram_shmidt(vectors, inner_prod):
    if len(vectors) == 1:
        return [(1.0/polynomial_norm(vectors[0], inner_prod))*vectors[0]]
    qs = gram_shmidt(vectors[:-1], inner_prod)
    vn = vectors[-1]
    qn = vn
    for q in qs:
        qn = qn - (vn.T @ inner_prod @ q)*q
    qs.append((1.0/polynomial_norm(qn, inner_prod))*qn)
    return qs


def polynomial_norm(v, inner_prod):
    return math.sqrt(v.T @ inner_prod @ v)


def polynomial_inner_prod_matrix(n, a, b):
    inner_prod = np.zeros(shape=[n, n])
    for i in range(n):
        for j in range(i, n):
            entry = polynomial_inner_prod_on_basis(a, b, i, j)
            inner_prod[i, j] = entry
            inner_prod[j, i] = entry
    return inner_prod


def polynomial_inner_prod_on_basis(a, b, i, j):
    power = i+j+1.0
    return (b**power - a**power)/power


def function_polynomial_inner_prod(a, b, f, pv):
    p = np.polynomial.Polynomial(pv)
    return integrate.quad(lambda x: f(x)*p(x), a, b)[0]


def output_for_desmos(pv):
    pl = pv.tolist()
    text = ""
    for (i, c) in enumerate(pl):
        text += str(c)+"x^"+str(i)
        if i < len(pl)-1:
            text += " + "
    return text


if __name__ == "__main__":
    main()
