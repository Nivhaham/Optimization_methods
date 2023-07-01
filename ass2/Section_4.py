import matplotlib.pyplot as plt
import numpy as np


def main():
    # Section A
    L = np.array([
        [2, -1, -1, 0, 0, 0, 0, 0, 0, 0],
        [-1, 2, -1, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, 3, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 5, -1, 0, -1, 0, 1, 1],
        [0, 0, 0, -1, 4, -1, -1, -1, 0, 0],
        [0, 0, 0, 0, -1, 3, -1, -1, 0, 0],
        [0, 0, 0, -1, -1, -1, 5, -1, 0, -1],
        [0, 0, 0, 0, -1, -1, -1, 4, 0, -1],
        [0, 0, 0, -1, 0, 0, 0, 0, 2, -1],
        [0, 0, 0, -1, 0, 0, -1, -1, -1, 4]
    ])

    b = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    b = b.transpose()
    print(L)
    building_graph(L, b)
    TEST=     L = np.array([
        [3,-1,-1,-1],
        [-1, 2, -1, 0],
        [-1, -1, 3, -1],
        [-1, 0, -1, 2]
    ])
    #print("\n TEST: \n",TEST,"\n")

    #L1=make_new_laplacian(TEST)
    #print(L1)
    # Section B
    #print("\n",make_laplican_block_matrix(1,3,L1))


def general_iterative_Iter(A, b, M, weight):
    graph_points = []
    graph_iteration_x = []
    x = np.zeros(A[0].size)
    for i in range(1000):
        x = x + weight * np.linalg.inv(M).dot(b - A.dot(x))
        residual_norm = np.linalg.norm(b - A @ x)
        graph_iteration_x.append((i + 1, x))
        graph_points.append((i + 1, residual_norm))
        if ((np.linalg.norm((A.dot(x) - b)) / (np.linalg.norm(b))) < 0.00001):
            # print("num of iteration is: \n" + str(i))
            # print(x)
            return graph_points, graph_iteration_x
            break

    # return x


def jacobi(A, b):
    M = np.eye(A[0].size)
    for i in range(A[0].size):
        M[i][i] = A[i][i]
    # N = A - M
    # x = np.zeros(A[0].size)
    return (general_iterative_Iter(A, b, M, 1))


def building_graph(A, b):
    # print(type(gmres(A,b)[0]))
    graph_points, graph_of_x = [i for i in jacobi(A, b)]
    # print(graph_points)
    x = [i[0] for i in graph_points]
    y = [i[1] for i in graph_points]
    print(x)
    print(y)
    convergence_rate = []
    x_star = graph_of_x[len(graph_of_x) - 1][1]
    for k in range(len(graph_of_x) - 2):
        convergence_rate.append(
            (k + 1, np.linalg.norm(graph_of_x[k + 1][1] - b) / (np.linalg.norm(graph_of_x[k][1] - b))))
    print(convergence_rate)

    plt.xlabel('iter num')
    plt.ylabel('residual')

    plt.title('Jacobi Method')

    plt.plot(x, y, label='Residual')
    plt.legend()
    plt.show()

    x2 = [i[0] for i in convergence_rate]
    y2 = [i[1] for i in convergence_rate]
    plt.plot(x2, y2, label='Residual rate')
    plt.legend()
    plt.show()





#Section B
ConvertArray=[1,2,0,3]
def make_new_laplacian(L):
    L_new=L
    for i in range(len(L[0])):
        for j in range(len(L[0])):
            #if i!=j:
            inew, jnew = convert_indexes(i,j)
            L_new[inew, jnew]=L[i,j]

    return(L_new)

def convert_indexes(i,j):
    newi=ConvertArray.index(i)
    newj=ConvertArray.index(j)
    return newi, newj


def make_laplican_block_matrix(i,j,L):
    M=L
    for i in range(len(L[0])):
        for j in range(len(L[0])):
            if ((i<2 and j>=2) or (i>=2 and j<2)):
                M[i][j]=0
    return M



if __name__ == '__main__':
    main()
