import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def main():
    A = np.array([[5, 4, 4, -1, 0],
                  [3, 12, 4, -5, -5],
                  [-4, 2, 6, 0, 3],
                  [4, 5, -7, 10, 2],
                  [1, 2, 5, 3, 10]])
    b = np.array([1, 1, 1, 1, 1])
    b = b.transpose()
    # print(A)
    # print(A@b)
    # print(gmres(A, b))
    building_graph(A, b)


def building_graph(A, b):
    # print(type(gmres(A,b)[0]))
    x = [i[0] for i in gmres(A, b)]
    y = [i[1] for i in gmres(A, b)]
    print(x)
    print(y)
    plt.xlabel('iter num')
    plt.ylabel('residual')

    plt.title('GMRES converge')

    #ay2.set_ylim([0.1, 0.000000001])

    plt.plot(x, y, label='converges rate')
    plt.legend()

    plt.show()


def gmres(A, b):
    graph_points = []
    x = np.zeros(A[0].size)
    r = b - A.dot(x)
    for i in range(50):
        # print(x)
        alpha = (r.transpose().dot(A).dot(r)) / (r.transpose().dot(A.transpose()).dot(A.dot(r)))
        x = x + alpha * r
        r = b - A.dot(x)
        r_norm = np.linalg.norm(r)
        # if r.transpose().dot(r) < 0.0001:
        graph_points.append((i + 1, r_norm))
        # if ((np.linalg.norm((A.dot(x) - b)) / (np.linalg.norm(b))) < 0.0000001):
        #   print("wow converged ")
        # return x
        # print("didn't converged yet in the iteration number \n" + str(i))
    return graph_points


# def gmres_2(A, b):
    # graph_points = []
    # x = np.zeros(A[0].size)
    # r = b - A.dot(x)
    # for i in range(50):
    #     a = [a1,a2]
    #     r_curr, r_prev
    #     alpha = (r.transpose().dot(A).dot(r)) / (r.transpose().dot(A.transpose()).dot(A.dot(r)))
    #     x = x + alpha * r
    #     r = b - A.dot(x)
    #     r_norm = np.linalg.norm(r)
    #     # if r.transpose().dot(r) < 0.0001:
    #     graph_points.append((i + 1, r_norm))
    #     # if ((np.linalg.norm((A.dot(x) - b)) / (np.linalg.norm(b))) < 0.0000001):
    #     #   print("wow converged ")
    #     # return x
    #     # print("didn't converged yet in the iteration number \n" + str(i))
    # return graph_points


if __name__ == '__main__':
    main()
