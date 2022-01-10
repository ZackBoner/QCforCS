import math
import re
from typing import Tuple


class Complex:
    def __init__(self, real=None, imaginary=None):
        if real is None:
            real = 0
        if imaginary is None:
            imaginary = 0

        self.real, self.imag = real, imaginary

    @classmethod
    def frominputstring(cls, input_string):
        real, imaginary = cls.parse_complex_num_string(input_string)
        if real is False and imaginary is False:
            raise Exception("Incorrect complex number format")
        return cls(int(real), int(imaginary))

    @classmethod
    def fromparameters(cls, real, imaginary):
        if not isinstance(real, (int, float)) or not isinstance(real, (int, float)):
            raise TypeError("Arguments must be of type `int`")

        return cls(real, imaginary)

    @classmethod
    def frompolar(cls, ro, theta):
        if not isinstance(ro, (int, float)) or not isinstance(theta, (int, float)):
            raise TypeError("Arguments must be of type `int`")

        real = ro * math.cos(theta)
        imaginary = ro * math.sin(theta)

        return cls(real, imaginary)

    def get_cartesian_coordinates(self):
        return self.real, self.imag

    def get_polar_coordinates(self):
        ro = math.sqrt(self.real ** 2 + self.imag ** 2)
        theta = math.atan2(self.real, self.imag)
        return ro, theta

    @staticmethod
    def parse_complex_num_string(input_string):
        p = re.compile(r"^(-?\d+)\s*([+\-])\s*(\d+)i$")
        input_string = input_string.strip()
        if (m := p.match(input_string)) is not None:
            real, op, imaginary = m.groups()
            return real, op + imaginary
        else:
            return False, False

    def __add__(self, other):
        try:
            other.real, other.imag
        except (AttributeError, TypeError):
            raise AssertionError("Argument must be of type `Complex`")

        real = self.real + other.real
        imaginary = self.imag + other.imag

        return self.fromparameters(real, imaginary)

    def __mul__(self, other):
        try:
            other.real, other.imag
        except (AttributeError, TypeError):
            raise AssertionError("Argument must be of type `Complex`")

        real = self.real * other.real - self.imag * other.imag
        imaginary = self.real * other.imag + self.imag * other.real

        return self.fromparameters(real, imaginary)

    def __truediv__(self, other):
        try:
            other.real, other.imag
        except (AttributeError, TypeError):
            raise AssertionError("Argument must be of type `Complex`")

        denominator = other.real ** 2 + other.imag ** 2
        real = (self.real * other.real + self.imag * other.imag) / denominator
        imaginary = (other.real * self.imag - self.real * other.imag) / denominator

        return self.fromparameters(real, imaginary)

    def __sub__(self, other):
        try:
            other.real, other.imag
        except (AttributeError, TypeError):
            raise AssertionError("Argument must be of type `Complex`")

        real = self.real - other.real
        imaginary = self.imag - other.imag

        return self.fromparameters(real, imaginary)

    def __abs__(self) -> float:
        return (self.real ** 2 + self.imag ** 2) ** (1 / 2)

    def modulus(self) -> float:
        return abs(self)

    def conjugate(self):
        return self.fromparameters(self.real, -1 * self.imag)

    def __pow__(self, power, modulo=None):
        if not isinstance(power, int):
            raise TypeError("Power must be an integer.")
        ro, theta = self.get_polar_coordinates()
        ro = ro ** power
        theta = power * theta
        return self.frompolar(ro, theta)

    def nth_root(self, n) -> Tuple[float, list]:
        if not isinstance(n, int):
            raise TypeError("To take nth root, pass in an integer n.")
        ro, theta = self.get_polar_coordinates()
        n_inv = 1 / n
        ro = ro ** n_inv
        roots = []
        for k in range(n):
            roots.append(n_inv * theta + k * n_inv * 2 * math.pi)
        return ro, roots

    def print_polar_coordinates(self):
        ro, theta = self.get_polar_coordinates()
        print(f"Polar Coordinates of ({self}):\np: {ro:.2f}\ntheta: {theta:.2f}")

    def __str__(self):
        return f"{self.real} + {self.imag}i"


if __name__ == '__main__':
    num1 = Complex.fromparameters(1, 2)
    num2 = Complex.fromparameters(3, 4)

    print(f"({num1}) + ({num2}) = {num1 + num2}")
    print(f"({num1}) * ({num2}) = {num1 * num2}")
    print(f"({num1}) - ({num2}) = {num1 - num2}")
    print(f"({num1}) / ({num2}) = {num1 / num2}")

    print()
    num3 = Complex.fromparameters(5, 6)
    print(f"Conjugate of ({num3}) = {num3.conjugate()}")
    print(f"|{num3}| = {abs(num3):.2f}")
    num3.print_polar_coordinates()

    num4 = Complex.fromparameters(1, -1)
    print(f"({num4})^5 = {num4 ** 5}")

    num5 = Complex.fromparameters(1, 1)
    print(f"({num5})^(1/3) = {num5.nth_root(3)}")
