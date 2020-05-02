import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt


def main():
    n = 10+1
    a = 0.1
    b = 10
    f = np.log

    inner_prod = polynomial_inner_prod_matrix(n, a, b)

    basis = gram_shmidt(
        standard_basis(n),
        inner_prod
    )

    projection = project(f, a, b, basis, inner_prod)
    # maclaurin = cosine_maclaurin(n)

    plot_comparison(f, {
        'Projection, n='+str(n-1): projection,
        # 'Maclaurin': maclaurin
    }, a, b)


def cosine_maclaurin(n):
    p = np.zeros(n)
    for i in range(math.ceil(n/2)):
        sign = (i % 2)*(-2)+1
        p[2*i] = sign/math.factorial(2*i)
    return p


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


def plot_comparison(f, poly_approx, a, b):
    x = np.linspace(a, b, 100)

    plt.plot(x, f(x), label='Original')
    lim = plt.ylim()

    for key in poly_approx:
        y = np.polynomial.Polynomial(poly_approx[key])
        plt.plot(x, y(x), label=key, linestyle='dashed')

    plt.ylim(lim)

    # plt.xlabel('x label')
    # plt.ylabel('y label')

    plt.title("Polynomial Fits")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
