import numpy as np


def dot(v1, v2):
    # v1 and v2 should be of type ComplexVector
    if not isinstance(v1, ComplexVector) or not isinstance(v2, ComplexVector):
        raise TypeError("v1 and v2 should be ComplexVector's")
    # the shapes should align
    if v1.shape()[1] != v2.shape()[0]:
        raise ValueError("Misaligned shapes for matrix multiplication.")

    return ComplexVector(np.dot(v1.vector, v2.vector))


def nvector_inner_product(v1, v2, debug=False):
    # v1 and v2 should be of type ComplexVector
    if not isinstance(v1, ComplexVector) or not isinstance(v2, ComplexVector):
        raise TypeError("v1 and v2 should be ComplexVector's")
    # v1 and v2 should also be column vectors
    if v1.shape()[1] != 1 or v2.shape()[1] != 1:
        raise ValueError("v1 or v2 is not a column vector")

    if debug:
        print(v1.shape(), v2.shape())
        print(v1.adjoint().shape())

    return dot(v1.adjoint(), v2).vector[0, 0]


def distance(v1, v2):
    # check type
    if not isinstance(v1, ComplexVector) or not isinstance(v2, ComplexVector):
        raise TypeError("v1 and v2 should be ComplexVector's")
    # check shape
    if v1.shape()[0] != v2.shape()[0] or v1.shape()[1] != v2.shape()[1]:
        raise ValueError("v1 and v2 must have the same shape")

    return np.sqrt(nvector_inner_product(v1 - v2, v1 - v2))


def tensor_product(v1, v2):
    # check type
    if not isinstance(v1, ComplexVector) or not isinstance(v2, ComplexVector):
        raise TypeError("v1 and v2 should be ComplexVector's")

    return ComplexVector(
        np.kron(v1.vector, v2.vector)
    )


class ComplexVector:
    def __init__(self, vector):
        if not isinstance(vector, (list, tuple, type(np.array([])))):
            raise TypeError("Invalid input to ComplexVector constructor.")
        self.vector = np.array(vector)

    def shape(self):
        return self.vector.shape

    def __len__(self):
        return len(self.vector)

    def __add__(self, other):
        if not isinstance(other, ComplexVector):
            raise TypeError("Must add ComplexVector objects.")
        return ComplexVector(np.add(self.vector, other.vector))

    def __mul__(self, other):
        if not isinstance(other, ComplexVector):
            raise TypeError("Must multiply ComplexVector objects.")
        return ComplexVector(np.multiply(self.vector, other.vector))

    def __neg__(self):
        return ComplexVector(self.vector * -1)

    def __sub__(self, other):
        if not isinstance(other, ComplexVector):
            raise TypeError("Must subtract ComplexVector objects.")
        return self + -other

    def norm(self, inner_product_func):
        return np.sqrt(inner_product_func(self, self))

    def transpose(self):
        return ComplexVector(np.transpose(self.vector))

    def conjugate(self):
        return ComplexVector(np.conjugate(self.vector))

    def adjoint(self):
        return self.conjugate().transpose()

    def is_hermitian(self):
        if self.shape()[0] != self.shape()[1]:
            raise ValueError("Matrix must be square to check if it is hermitian.")
        return np.array_equal(self.adjoint().vector, self.vector)

    def is_unitary(self):
        if self.shape()[0] != self.shape()[1]:
            raise ValueError("Matrix must be square to check if it is unitary.")
        n = self.shape()[0]
        identity_matrix = np.identity(n)
        # an nxn matrix U is unitary if U*adjoint(U) = adjoint(U)*U = In
        return np.array_equal((self * self.adjoint()).vector, identity_matrix) and np.array_equal(
            (self.adjoint() * self).vector, identity_matrix)

    def __str__(self):
        return str(self.vector)


if __name__ == '__main__':
    m, n = int(input("Enter number of rows.")), int(input("Enter number of columns."))

    c1 = list()
    for rdx in range(m):
        row = []
        for cdx in range(n):
            print(f"c1[{rdx}, {cdx}]")
            row.append(complex(input()))
        c1.append(row)
    c1 = np.array(c1)
    print("c1:")
    print(c1)
    print()

    j, k = int(input("Enter number of rows.")), int(input("Enter number of columns."))
    c2 = list()
    for rdx in range(j):
        row = []
        for cdx in range(k):
            print(f"c2[{rdx}, {cdx}]")
            row.append(complex(input()))
        c2.append(row)
    c2 = np.array(c2)
    print("c2:")
    print(c2)
    print()

    c1, c2 = ComplexVector(c1), ComplexVector(c2)

    print("c1 + c2:")
    if j == m and k == n:
        print(c1 + c2)
    print()
    print("Negative c1 and c2")
    print(-c1)
    print(-c2)
    print()
    if n == k:
        print("c1 * c2, adjoint of (c1 * c2), and (adjoint of c2) * (adjoint of c1)")
        print(c1 * c2)
        print()
        print((c1 * c2).adjoint())
        print(c2.adjoint() * c1.adjoint())
    print()

    print("Inner product of c1 and c2 (only for nx1 vectors)")
    try:
        print(nvector_inner_product(c1, c2))
    except ValueError:
        print("Invalid shapes.")

    print("Norms and distance of c1 and c2")
    print()
    try:
        print(c1.norm(nvector_inner_product))
        print(c2.norm(nvector_inner_product))
        print(distance(c1, c2))
    except ValueError:
        print("c1 or c2 is not a column vector")
    print()

    print("Test if c1 and c2 are hermitian.")
    try:
        print(c1.is_hermitian())
        print(c2.is_hermitian())
    except ValueError:
        print("c1 or c2 was not square")

    print("Test if c1 and c2 are unitary.")
    try:
        print(c1.is_unitary())
        print(c2.is_unitary())
    except ValueError:
        print("c1 or c2 was not square")
    print()

    print("Tensor product (both ways) of c1 and c2")
    print(tensor_product(c1, c2))
    print()
    print(tensor_product(c2, c1))
    print()
