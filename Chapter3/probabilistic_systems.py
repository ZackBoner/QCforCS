import numpy as np
np.set_printoptions(precision=3, linewidth=np.inf, suppress=True)


class System:
    def __init__(self, matrix):
        self.matrix = np.array(matrix)
        self.n = self.matrix.shape[0]

    @classmethod
    def frominput(cls):
        n = input("Number of states: ")
        matrix = np.zeros((n, n))
        print("Input matrix:")
        for i in range(n):
            row = input().split(" ")
            row = list(map(complex, row))
            matrix[i] = row
        return cls(matrix)

    def simulate(self, initial, n_steps: int):
        return np.linalg.matrix_power(self.matrix, n_steps) @ initial


def bullet_experiment(n_slits: int, n_targets: int, P_slit_to_target: dict, PRINT: bool = False):
    n = n_slits + n_targets + 1
    B = np.zeros((n, n))
    B[1:1+n_slits, 0] = 1/n_slits
    B[-n_targets:, -n_targets:] = np.identity(n_targets)
    for slit in P_slit_to_target:
        B[-n_targets:, slit] = P_slit_to_target[slit]

    system = System(B)

    if PRINT:
        initial = np.zeros(n)
        initial[0] = 1
        print(B)
        print(np.linalg.matrix_power(B, 2))
        print(system.simulate(initial, 2))

    return system


def photon_experiment(n_slits: int, n_targets: int, P_slit_to_target: dict, PRINT: bool = False):
    n = n_slits + n_targets + 1
    B = np.zeros((n, n), dtype=complex)
    B[1:1+n_slits, 0] = 1/np.sqrt(n_slits)
    B[-n_targets:, -n_targets:] = np.identity(n_targets)
    for slit in P_slit_to_target:
        B[-n_targets:, slit] = P_slit_to_target[slit]

    system = System(B)

    if PRINT:
        initial = np.zeros(n)
        initial[0] = 1
        print(B)
        print(np.linalg.matrix_power(B, 2))
        print(system.simulate(initial, 2))

    return system


if __name__ == "__main__":
    system = System([[1/2, 0, 1/2],
                    [0, 1/2, 1/2],
                    [1/2, 1/2, 0]])
    print(system.simulate([1/5, 3/5, 1/5], 100))

    print("Bullet experiment:")
    b_system = bullet_experiment(2,
                      5,
                      {1: [1/3, 1/3, 1/3, 0, 0], 2: [0, 0, 1/3, 1/3, 1/3]},
                      PRINT=True)
    print()
    print("Photon experiment:")
    p_system = photon_experiment(2,
                      5,
                      {1: [complex(-1, 1)/np.sqrt(6), complex(-1, -1)/np.sqrt(6), complex(1, -1)/np.sqrt(6), 0, 0],
                       2: [0, 0, complex(-1, 1)/np.sqrt(6), complex(-1, -1)/np.sqrt(6), complex(1, -1)/np.sqrt(6)]},
                      PRINT=True)
    
    B_2 = np.linalg.matrix_power(b_system.matrix, 2)
    P_2 = np.linalg.matrix_power(p_system.matrix, 2)
    P_2_mod = abs(P_2)**2

    print()
    print("Interference detection:")
    interference = np.ceil(B_2 - P_2_mod).astype(int)
    print(interference)
    print((interference != 0).nonzero())


    

