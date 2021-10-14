import matplotlib.pyplot as plt
from complex_numbers import Complex


def make_ints(li):
    return tuple(int(x) for x in li)


def complex_multiply(points, c):
    complex_points = []
    for point in points:
        complex_points.append((point[0]*c.real-point[1]*c.imag,
                               point[1]*c.real+point[0]*c.imag))
    return complex_points


if __name__ == "__main__":
    pass
