import numpy as np
from scipy.sparse import random
import scipy.sparse as sparse
import matplotlib.pyplot as plt


# Question 1
def general_iterative_Iter(A, x, b, M, N, weight):
    building_convergence_graph = []
    for i in range(100):
        x = x + weight * np.linalg.inv(M).dot(b - A.dot(x))
        building_convergence_graph.append((i + 1, np.linalg.norm((A.dot(x) - b))))
        # print(np.linalg.norm((A.dot(x) - b)))
        # print(np.linalg.norm((A.dot(x) - b)) / (np.linalg.norm(b)))
        # if ((np.linalg.norm((A.dot(x) - b)) / (np.linalg.norm(b))) < 0.1):
        #    pass
    return building_convergence_graph


def jacobi(A, b):
    M = np.eye(A[0].size)
    for i in range(A[0].size):
        M[i][i] = A[i][i]
    N = A - M
    x = np.zeros(A[0].size)
    return (general_iterative_Iter(A, x, b, M, N,0.36))


def gauss_Seid(A, b):
    M = np.eye(A[0].size)
    for i in range(A[0].size):
        for j in range(A[0].size):
            if (j >= i):
                M[i][j] = A[i][j]
    N = A - M
    x = np.zeros(A[0].size)
    return (general_iterative_Iter(A, x, b, M, N, 1))


def gradient_descent(A, b):
    x = np.zeros(A[0].size)
    building_convergence_graph = []
    r = b - A.dot(x)
    for i in range(100):
        # print(x)
        alpha = (r.transpose().dot(r)) / (r.transpose().dot(A.dot(r)))
        x = x + alpha * r
        r = b - A.dot(x)
        building_convergence_graph.append((i + 1, np.linalg.norm(r)))
        # if r.transpose().dot(r) < 0.0001:
        # if ((np.linalg.norm((A.dot(x) - b)) / (np.linalg.norm(b))) < 0.001):
        # pass
    return building_convergence_graph


def conjugate_Gradient(A, b):
    x = np.zeros(A[0].size)
    building_convergence_graph = []
    r = b - A.dot(x)
    p = r
    for i in range(100):
        alpha = (r.transpose().dot(r)) / (p.transpose().dot(A.dot(p)))
        x_old = x
        x = x + alpha * p
        old_r = r
        r = b - A.dot(x)
        # if r.transpose().dot(r) < 0.0001:
        beta = (r.transpose().dot(r)) / (old_r.transpose().dot(old_r))
        p = r + beta * p
        building_convergence_graph.append((i + 1, np.linalg.norm(r)))
    return building_convergence_graph


def graphs(A, b):
    graph_points_jacobi = jacobi(A, b)
    graph_points_gauss_Seid = gauss_Seid(A, b)
    graph_points_gradient_descent = gradient_descent(A, b)
    graph_points_conjugate_Gradient = conjugate_Gradient(A, b)

    x = [i + 1 for i in range(100)]

    y_jacobi = [i[1] for i in graph_points_jacobi]
    build_graph_residual(x, y_jacobi, "Jacobi method residual")
    y_gauss_Seid = [i[1] for i in graph_points_gauss_Seid]
    build_graph_residual(x, y_gauss_Seid, "gauss_Seid method residual")
    y_gradient_descent = [i[1] for i in graph_points_gradient_descent]
    build_graph_residual(x, y_gradient_descent, "gradient_descent method residual")
    y_conjugate_Gradient = [i[1] for i in graph_points_conjugate_Gradient]
    build_graph_residual(x, y_conjugate_Gradient, "conjugate_Gradient method residual")

    build_graph_convergence_rate(x, y_jacobi, "Jacobi method convergence rate")
    build_graph_convergence_rate(x, y_gauss_Seid, "gauss_Seid_residual method convergence rate")
    build_graph_convergence_rate(x, y_gradient_descent, "gradient_descent method convergence rate")
    build_graph_convergence_rate(x, y_conjugate_Gradient, "conjugate_Gradient method convergence rate")


def build_graph_residual(x, y, name):
    plt.xlabel('iter num')
    plt.ylabel('residual')

    plt.title(name)

    plt.semilogy(x, y, label='Residual')
    plt.legend()
    plt.show()
    pass


def build_graph_convergence_rate(x, y, name):
    plt.xlabel('iter num')
    plt.ylabel('ratio')

    plt.title(name)

    for i in range(len(y) - 1):
        y[i] = y[i + 1] / y[i]
    y[len(y)-1] = y[len(y)-2]
    plt.plot(x, y, label='Convergence rate')
    plt.legend()
    plt.show()


def main():
    n = 256
    A = random(n, n, 5 / n, dtype=float)
    v = np.random.rand(n)
    v = sparse.spdiags(v, 0, v.shape[0], v.shape[0], 'csr')
    A = A.transpose() * v * A + 0.1 * sparse.eye(n)
    A = A.toarray()
    b = np.random.rand(256, 1)
    # print(gauss_Seid(A,b))
    # print(gradient_descent(A,b))
    # print(conjugate_Gradient(A,b))
    graphs(A, b)


if __name__ == '__main__':
    main()

# #Question 3 Section C
# def gradient_descent(A, b):
#     x = np.zeros(A[0].size)
#     r = b - A.dot(x)
#     for i in range(100):
#         print(x)
#         alpha = (r.transpose().dot(r)) / (r.transpose().dot(A.dot(r)))
#         x = x + alpha * r
#         r = b - A.dot(x)
#         # if r.transpose().dot(r) < 0.0001:
#         if ((np.linalg.norm((A.dot(x) - b)) / (np.linalg.norm(b))) < 0.001):
#             return x
#             break
# check for question 1.a
# A = np.array([[2, 1, 2], [1, 7, 1], [1, 2, 3]])
# AA=A.transpose()@A
# print(AA@np.array([1,2,3]))
# b = np.array([10, 18, 14])
# bb= np.array([52, 164, 80])
#
