
from qiskit import QuantumRegister, QuantumCircuit 
from qiskit.circuit.library import RYGate, IntegerComparator

import numpy as np

def _large_coefficients_iter(m, n):
    """Return an iterator over multinomial coefficients"""
    if m < 2 * n or n == 1:
        coefficients = _multinomial_coefficients(m, n)
        for key, value in coefficients.items():
            yield (key, value)
    else:
        coefficients = _multinomial_coefficients(n, n)
        coefficients_dict = {}
        for key, value in coefficients.items():
            coefficients_dict[tuple(filter(None, key))] = value
        coefficients = coefficients_dict

        temp = [n] + [0] * (m - 1)
        temp_a = tuple(temp)
        b = tuple(filter(None, temp_a))
        yield (temp_a, coefficients[b])
        if n:
            j = 0  # j will be the leftmost nonzero position
        else:
            j = m
        # enumerate tuples in co-lex order
        while j < m - 1:
            # compute next tuple
            temp_j = temp[j]
            if j:
                temp[j] = 0
                temp[0] = temp_j
            if temp_j > 1:
                temp[j + 1] += 1
                j = 0
            else:
                j += 1
                temp[j] += 1

            temp[0] -= 1
            temp_a = tuple(temp)
            b = tuple(filter(None, temp_a))
            yield (temp_a, coefficients[b])


def _binomial_coefficients(n):
    """Return a dictionary of binomial coefficients"""
    data = {(0, n): 1, (n, 0): 1}
    temp = 1
    for k in range(1, n // 2 + 1):
        temp = (temp * (n - k + 1)) // k
        data[k, n - k] = data[n - k, k] = temp
    return data


def _multinomial_coefficients(m, n):
    """Return the multinomial coefficients as a dictionary"""
    if not m:
        if n:
            return {}
        return {(): 1}
    if m == 2:
        return _binomial_coefficients(n)
    if m >= 2 * n and n > 1:
        return dict(_large_coefficients_iter(m, n))
    if n:
        j = 0
    else:
        j = m
    temp = [n] + [0] * (m - 1)
    res = {tuple(temp): 1}
    while j < m - 1:
        temp_j = temp[j]
        if j:
            temp[j] = 0
            temp[0] = temp_j
        if temp_j > 1:
            temp[j + 1] += 1
            j = 0
            start = 1
            v = 0
        else:
            j += 1
            start = j + 1
            v = res[tuple(temp)]
            temp[j] += 1
        for k in range(start, m):
            if temp[k]:
                temp[k] -= 1
                v += res[tuple(temp)]
                temp[k] += 1
        temp[0] -= 1
        res[tuple(temp)] = (v * temp_j) // (n - temp[0])
    return res


def _univariate_monomial(num_qubits_half, max_univariate_degree):
    """Return a dictionary of univariate monomial coefficients for argument expressed as a binary"""
    x = np.linspace(0, num_qubits_half-1, num_qubits_half)

    dec = lambda t: 2**t
    x = dec(x)[::-1]

    coeffs = {}

    for t, factor in _multinomial_coefficients(
        num_qubits_half, max_univariate_degree
    ).items():
        facp = 1.0
        tt = []
        for i, tac in enumerate(t):
            if tac > 0:
                facp *= x[i] ** tac
                tt.append(int(1))
            else:
                tt.append(0)

        prefac = coeffs.get(tuple(tt), 0.0)

        coeffs[tuple(tt)] = prefac + factor * facp

    return coeffs


class BivariatePolynomialEncoder:
    """Main class for generating the quantum circuit for a given bivariate polynomial."""

    def __init__(
        self,
        poly_coeffs,
        wires,
        num_qubits_half,
        max_univariate_degree,
        eps,
        id=None,
    ):
        self.poly_coeffs = poly_coeffs
        self.wires = wires
        self.num_qubits_half = num_qubits_half
        self.max_univariate_degree = max_univariate_degree
        self.eps = eps
        self.id = id

    def build(self):
        """
        Constructs the quantum operations for the multivariate polynomial encoding.

        Args:
            wires (list): List of qubit indices to apply the operation.

        Returns:
            QuantumCircuit: The quantum circuit with the decomposition applied.
        """
        coeffs_new = self.poly_coeffs * self.eps
        qubit_coeffs_xy = {}

        # Calculate the decomposition
        for i_deg in range(self.max_univariate_degree + 1):
            dec_coeffs_x = _univariate_monomial(self.num_qubits_half, i_deg)

            for j_deg in range(self.max_univariate_degree + 1):
                dec_coeffs_y = _univariate_monomial(self.num_qubits_half, j_deg)

                poly_coeffs_ij = coeffs_new[i_deg][j_deg]
                if poly_coeffs_ij == 0.0:
                    continue

                for key_x, value_x in dec_coeffs_x.items():
                    for key_y, value_y in dec_coeffs_y.items():
                        tt = list(key_x) + list(key_y)
                        factor = 2 * value_x * value_y

                        prefac = qubit_coeffs_xy.get(tuple(tt), 0.0)
                        qubit_coeffs_xy[tuple(tt)] = prefac + factor * poly_coeffs_ij

        coeffs_list = list(qubit_coeffs_xy.values())

        # Create a QuantumCircuit and apply the controlled RY rotations
        circuit = QuantumCircuit(len(self.wires), name="multi-poly")
        for j, c in enumerate(qubit_coeffs_xy.keys()):
            qr_ctrl = []

            for i, _ in enumerate(c):
                if c[i] > 0:
                    # qr_wires.append(num_qubits_half*2-(i))
                    if self.wires[i] >= self.num_qubits_half:
                        qr_ctrl.append((3 * self.num_qubits_half) - self.wires[i] - 1)
                    else:
                        qr_ctrl.append(self.num_qubits_half - self.wires[i] - 1)

            # Apply controlled-RY gate
            if len(qr_ctrl) == 0:
                circuit.ry(coeffs_list[j], self.wires[-1])
            else:
                # Use RYGate and apply control
                rotation_angle = coeffs_list[j]
                ry_gate = RYGate(rotation_angle)

                # Create the controlled version of the RY gate
                controlled_ry = ry_gate.control(len(qr_ctrl))

                # Append the controlled gate to the circuit
                circuit.append(controlled_ry, qr_ctrl + [self.wires[-1]])

        return circuit

class BivarPolyLimit:
    """
    coeffs_matrix: 2D numpy array of coefficients of the bivariate polynomial
    limit: the limit(boundary) of the bivariate polynomial
    num_qubits_half: number of qubits for each variable
    max_univariate_degree: maximum degree of the univariate polynomial
    eps: precision factor for the encoding
    """
    def __init__(
        self,
        coeffs_matrix,
        limit,
        num_qubits_half,
        max_univariate_degree, 
        eps,
        name = "BivarPolyLimit"
    ):
        
        self.coeffs_matrix = coeffs_matrix
        self.limit = limit
        self.num_qubits_half = num_qubits_half
        self.max_univariate_degree = max_univariate_degree
        self.eps = eps
        self.name = name

    def build(self):

        x_wires = [i for i in range(self.num_qubits_half)]
        y_wires = [i for i in range(self.num_qubits_half, 2*self.num_qubits_half)]
        wires_all = x_wires + y_wires + [2*self.num_qubits_half]

        qr = QuantumRegister(3*self.num_qubits_half+1)
        qc = QuantumCircuit(qr, name=self.name) 

        qc.append(IntegerComparator(num_state_qubits=self.num_qubits_half, value=int(self.limit), geq = True), x_wires + [2*self.num_qubits_half+1] + [i for i in range(2*self.num_qubits_half+2, 2*self.num_qubits_half+2+self.num_qubits_half-1)]) 
        qc.append(IntegerComparator(num_state_qubits=self.num_qubits_half, value=int(self.limit), geq = True), y_wires + [2*self.num_qubits_half+1] + [i for i in range(2*self.num_qubits_half+2, 2*self.num_qubits_half+2+self.num_qubits_half-1)]) 
        multi_poly = BivariatePolynomialEncoder(self.coeffs_matrix, wires_all, self.num_qubits_half, self.max_univariate_degree, self.eps).build().to_gate()
        qc.append(multi_poly, wires_all)

        return qc


class BivarPoly:
    """Piecewise polynomial 2D function."""

    def __init__(
        self,
        num_qubits_half: int,
        coeffs_matrix: np.ndarray,
        max_univariate_degree: int,
        breakpoints: np.ndarray | list,
        eps: float = 1e-6,
        name="piecewise_poly",
    ) -> None:
        """
        Args:
        num_qubits_half (int): Number of qubits to encode each variable.
        coeffs_matrix (ndarray): 4D array of coefficients with shape (num_qubits_half, num_qubits_half,
                                max_univariate_degree+1, max_univariate_degree+1), where coeffs_matrix[i][j]
                                is a 2D array of coefficients for the polynomial in the (i, j) patch.
        max_univariate_degree (int): Maximum degree of the univariate polynomials.
        breakpoints (ndarray or list): 2D array  of breakpoints for the piecewise polynomial. The breakpoints for the
                                i-th variable are given by breakpoints[i]. The breakpoints are assumed to be integers
                                and are sorted in increasing order. The number of breakpoints for each variable is
                                equal to the number of patches. Assume the first breakpoint is 0 and the last breakpoint
                                is N-1, where N is the number of qubits per variable.
        eps (float):  Scaling parameter (used for linearisation).
        """
        self.num_qubits_half = num_qubits_half
        self.coeffs_matrix = coeffs_matrix
        self.max_univariate_degree = max_univariate_degree
        self.breakpoints = breakpoints
        self.eps = eps
        self.name = name

    def build(self):
        coeffs_matrix_enlarged = np.zeros((
            len(self.breakpoints) + 1,
            len(self.breakpoints) + 1,
            self.max_univariate_degree + 1,
            self.max_univariate_degree + 1,
        ))
        coeffs_matrix_enlarged[1:, 1:, :, :] = self.coeffs_matrix
        coeffs_matrix_mapped = (
            self.coeffs_matrix
            - coeffs_matrix_enlarged[:-1, 1:, :, :]
            - coeffs_matrix_enlarged[1:, :-1, :, :]
            + coeffs_matrix_enlarged[:-1, :-1, :, :]
        )

        x_wires = [i for i in range(self.num_qubits_half)]
        y_wires = [i for i in range(self.num_qubits_half, 2 * self.num_qubits_half)]
        wires_all = x_wires + y_wires + [2 * self.num_qubits_half]

        qr = QuantumRegister(2 * self.num_qubits_half + 3, name="qr_piecewisepoly")

        qr_ancilla = QuantumRegister(
            self.num_qubits_half - 1, name="ancilla_piecewisepoly"
        )

        qc = QuantumCircuit(qr, qr_ancilla, name=self.name)

        # Apply polynomial (0, 0)
        qc.append(
            BivariatePolynomialEncoder(
                coeffs_matrix_mapped[0][0],
                wires_all,
                self.num_qubits_half,
                self.max_univariate_degree,
                self.eps,
            )
            .build()
            .to_gate(),
            wires_all,
        )

        # Apply polynomial (0, j)
        for j in range(1, len(self.breakpoints)):
            qc.append(
                IntegerComparator(
                    num_state_qubits=self.num_qubits_half,
                    value=int(self.breakpoints[1][j]),
                    geq=True,
                ),
                y_wires
                + [2 * self.num_qubits_half + 2]
                + [
                    i
                    for i in range(
                        2 * self.num_qubits_half + 3,
                        2 * self.num_qubits_half + 3 + self.num_qubits_half - 1,
                    )
                ],
            )
            bivar_poly_gate = (
                BivariatePolynomialEncoder(
                    coeffs_matrix_mapped[0][j],
                    wires_all,
                    self.num_qubits_half,
                    self.max_univariate_degree,
                    self.eps,
                )
                .build()
                .to_gate()
            )
            cpoly = bivar_poly_gate.control(1, label=None)
            qc.append(cpoly, [2 * self.num_qubits_half + 2] + wires_all)
            qc.append(
                IntegerComparator(
                    num_state_qubits=self.num_qubits_half,
                    value=int(self.breakpoints[1][j]),
                    geq=True,
                ).inverse(),
                y_wires
                + [2 * self.num_qubits_half + 2]
                + [
                    i
                    for i in range(
                        2 * self.num_qubits_half + 3,
                        2 * self.num_qubits_half + 3 + self.num_qubits_half - 1,
                    )
                ],
            )

        # Apply polynomial (i, 0)
        for i in range(1, len(self.breakpoints)):
            qc.append(
                IntegerComparator(
                    num_state_qubits=self.num_qubits_half,
                    value=int(self.breakpoints[0][i]),
                    geq=True,
                ),
                x_wires
                + [2 * self.num_qubits_half + 1]
                + [
                    i
                    for i in range(
                        2 * self.num_qubits_half + 3,
                        2 * self.num_qubits_half + 3 + self.num_qubits_half - 1,
                    )
                ],
            )
            bivar_poly_gate = (
                BivariatePolynomialEncoder(
                    coeffs_matrix_mapped[i][0],
                    wires_all,
                    self.num_qubits_half,
                    self.max_univariate_degree,
                    self.eps,
                )
                .build()
                .to_gate()
            )
            cpoly = bivar_poly_gate.control(1, label=None)
            qc.append(cpoly, [2 * self.num_qubits_half + 1] + wires_all)
            qc.append(
                IntegerComparator(
                    num_state_qubits=self.num_qubits_half,
                    value=int(self.breakpoints[0][i]),
                    geq=True,
                ).inverse(),
                x_wires
                + [2 * self.num_qubits_half + 1]
                + [
                    i
                    for i in range(
                        2 * self.num_qubits_half + 3,
                        2 * self.num_qubits_half + 3 + self.num_qubits_half - 1,
                    )
                ],
            )

        # Apply polynomial (i, j)
        for i in range(1, len(self.breakpoints)):
            for j in range(1, len(self.breakpoints)):
                qc.append(
                    IntegerComparator(
                        num_state_qubits=self.num_qubits_half,
                        value=int(self.breakpoints[0][i]),
                        geq=True,
                    ),
                    x_wires
                    + [2 * self.num_qubits_half + 1]
                    + [
                        i
                        for i in range(
                            2 * self.num_qubits_half + 3,
                            2 * self.num_qubits_half + 3 + self.num_qubits_half - 1,
                        )
                    ],
                )
                qc.append(
                    IntegerComparator(
                        num_state_qubits=self.num_qubits_half,
                        value=int(self.breakpoints[1][j]),
                        geq=True,
                    ),
                    y_wires
                    + [2 * self.num_qubits_half + 2]
                    + [
                        i
                        for i in range(
                            2 * self.num_qubits_half + 3,
                            2 * self.num_qubits_half + 3 + self.num_qubits_half - 1,
                        )
                    ],
                )
                bivar_poly_gate = (
                    BivariatePolynomialEncoder(
                        coeffs_matrix_mapped[i][j],
                        wires_all,
                        self.num_qubits_half,
                        self.max_univariate_degree,
                        self.eps,
                    )
                    .build()
                    .to_gate()
                )
                ccpoly = bivar_poly_gate.control(2, label=None)
                qc.append(
                    ccpoly,
                    [2 * self.num_qubits_half + 1, 2 * self.num_qubits_half + 2]
                    + wires_all,
                )
                qc.append(
                    IntegerComparator(
                        num_state_qubits=self.num_qubits_half,
                        value=int(self.breakpoints[1][j]),
                        geq=True,
                    ).inverse(),
                    x_wires
                    + [2 * self.num_qubits_half + 1]
                    + [
                        i
                        for i in range(
                            2 * self.num_qubits_half + 3,
                            2 * self.num_qubits_half + 3 + self.num_qubits_half - 1,
                        )
                    ],
                )
                qc.append(
                    IntegerComparator(
                        num_state_qubits=self.num_qubits_half,
                        value=int(self.breakpoints[0][i]),
                        geq=True,
                    ).inverse(),
                    y_wires
                    + [2 * self.num_qubits_half + 2]
                    + [
                        i
                        for i in range(
                            2 * self.num_qubits_half + 3,
                            2 * self.num_qubits_half + 3 + self.num_qubits_half - 1,
                        )
                    ],
                )

        return qc
