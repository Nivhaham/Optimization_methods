import numpy as np
import math
import time


def main():
    # Question 5 Seqtion b
    epsilon = 1
    A = np.array([[1, 1, 1], [epsilon, 0, 0], [0, epsilon, 0], [0, 0, epsilon]])
    Q1, R1 = gram_schmedit_version_1(A)
    Q2, R2 = gram_schmedit_version_2(A)
    print("gram schmedit version 1 for epsilon equals 1: \n\n", Q1, "\n\n", R1)
    print("\ngram schmedit version 2 for epsilon equals 1: \n\n", Q2, "\n\n", R2)
    
    # print("For the matrix with the epsilon value of 1, the first version of QR factorization leasted ",(end1-start), " seconds, while the second version leasted only ",(end2-start)," seconds.")

    epsilon = pow((math.e), -10)
    A = np.array([[1, 1, 1], [epsilon, 0, 0], [0, epsilon, 0], [0, 0, epsilon]])
    # print(get_vector(1, A))

    Q3, R3 = gram_schmedit_version_1(A)
    Q4, R4 = gram_schmedit_version_2(A)
    print("\n\n\n", "gram schmedit version 1 for epsilon equals e^-10: \n\n", Q3, "\n\n", R3)
    print("\n gram schmedit version 2 for epsilon equals e^-10: \n\n", Q4, "\n\n", R4)

    print(
        "\n\n\nSection C: \nFor epsilon = 1 with gram schmedit version 1, the Frobenius norm of the matrix QtQ-In is: ")
    frobenius(Q1)
    print("\n\nFor epsilon = 1 with gram schmedit version 2, the Frobenius norm of the matrix QtQ-In is: ")
    frobenius(Q2)
    print("\n\nFor epsilon = e^-10 with gram schmedit version 1, the Frobenius norm of the matrix QtQ-In is: ")
    frobenius(Q3)
    print("\n\nFor epsilon = e^-10 with gram schmedit version 2, the Frobenius norm of the matrix QtQ-In is: ")
    frobenius(Q4)
    print(
        "\nWe noticed that for both cases, the second version of gram schmedit retrieved us better values of numeric zero")


def frobenius(Mat):
    M = np.array(Mat)
    MtM = M.transpose() @ M
    print(np.linalg.norm(MtM - np.eye(len(MtM))))


def get_vector(j, mat):
    return np.array(mat[:, j - 1])


# Section A
def gram_schmedit_version_1(mat):
    R = np.zeros((len(mat[0]), len(mat[0])))
    a1 = get_vector(1, mat)
    a1_norm = np.linalg.norm(a1)
    R[0, 0] = a1_norm
    q1 = a1 / a1_norm
    Q = q1.reshape(4, 1)

    for i in range(2, len(mat[0]) + 1):
        q = get_vector(i, mat)
        for j in range(1, i):
            R[j - 1, i - 1] = get_vector(j, Q).transpose().dot(get_vector(i, mat))
            q = q - R[j - 1, i - 1] * get_vector(j, Q)

        R[i - 1, i - 1] = np.linalg.norm(q)
        q = q / R[i - 1, i - 1]
        Q = np.c_[Q, q]

    return Q, R


# Section B
def gram_schmedit_version_2(mat):
    R = np.zeros((len(mat[0]), len(mat[0])))
    a1 = get_vector(1, mat)
    a1_norm = np.linalg.norm(a1)
    R[0, 0] = a1_norm
    q1 = a1 / a1_norm
    Q = q1.reshape(4, 1)

    for i in range(2, len(mat[0]) + 1):
        q = get_vector(i, mat)
        for j in range(1, i):
            R[j - 1, i - 1] = get_vector(j, Q).transpose().dot(q)
            q = q - R[j - 1, i - 1] * get_vector(j, Q)

        R[i - 1, i - 1] = np.linalg.norm(q)
        q = q / R[i - 1, i - 1]
        Q = np.c_[Q, q]

    return Q, R


if __name__ == '__main__':
    main()
