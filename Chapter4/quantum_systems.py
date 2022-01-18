import numpy as np
from Chapter2.complex_vector_spaces import ComplexVector, nvector_inner_product, dot
import warnings


class Ket:
    def __init__(self, amplitudes):
        self.amplitudes: ComplexVector = ComplexVector(amplitudes).as_column_vector()

    def p(self, state: int) -> float:
        return (abs(self.amplitudes[state])**2/self.amplitudes.norm())[0]
    
    def p_transition(self, ket2: ComplexVector) -> float:
        return abs(nvector_inner_product(self.amplitudes, ket2.amplitudes))/(self.amplitudes.norm() * ket2.amplitudes.norm())


class Observable:
    def __init__(self, observable: ComplexVector, state: Ket):
        if not observable.is_hermitian():
            raise ValueError("Observable must be hermitian.")
        self.observable = observable
        self.state = Ket(state.amplitudes.as_column_vector())
    
    def mean(self) -> ComplexVector:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mean = float(nvector_inner_product(dot(self.observable, self.state.amplitudes), self.state.amplitudes))
        return mean

    def variance(self) -> ComplexVector:
        delta_phi = self.observable - ComplexVector(self.mean()*np.identity(self.observable.shape[0]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            var = float(dot(dot(self.state.amplitudes.adjoint(), dot(delta_phi, delta_phi)), self.state.amplitudes)[0, 0])
        return var


if __name__ == '__main__':
    ket = Ket([[complex(0, 1)], [complex(-1, 0)]])
    ket2 = Ket([[complex(1, 0)], [complex(0, -1)]])
    print(ket.p(0))
    print(ket.p(1))
    print(ket2.p(0))
    print(ket2.p(1))
    print(ket.p_transition(ket2))
    print(ket2.p_transition(ket))
    print()

    observable = ComplexVector([[1, complex(0, -1)], [complex(0, 1), 2]])
    ket = Ket([np.sqrt(2)/2, complex(0, np.sqrt(2)/2)])
    system = Observable(observable, ket)
    print(f"Mean: {system.mean(): 0.2f}")
    print(f"Variance: {system.variance()}")