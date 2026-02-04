# Quantum Spectral Method (QSM) for non-periodic boundary value problems

A reproducible Python/Qiskit implementation of the examples in [A Quantum Spectral Method for Non-Periodic Boundary Value Problems, E. Febrianto, Y. Wang, B. Liu, M. Ortiz and F. Cirak (2025)](https://arxiv.org/abs/2511.11494). 

---

## Overview

This repository provides a reference implementation of the **Quantum Spectral Method (QSM)** for **non-periodic boundary value problems**.
The core idea is to handle non-periodic boundary conditions via **domain extension** (e.g., odd/antisymmetric extension), enabling Fourier-based spectral operators to be implemented (or approximated) using quantum circuit primitives.
The implementation uses **Qiskit**.

---

## User setup

    conda env create -f environment.yml
    conda activate myqiskit
    jupyter notebook

---

## Example: One-dimensional Poisson–Dirichlet problem (homogeneous BC)

### Problem definition

We consider on the domain $\Omega = (0, 1) \subset ℝ^1$ the Poisson equation

$$
-\frac{\mathrm{d}^2 u(x)}{\mathrm{d}x^2} = f(x), \quad \forall x \in \Omega,
$$

with the homogeneous Dirichlet boundary conditions $u(0) = u(1) = 0$.

### Classical solution through antisymmetric reflection and DFT

We extend the physical fields, $f(x)$ and $u(x)$, to the domain $\Omega_{\mathrm{E}} = (0, 2)$. For the force $f(x)$:

$$
f_{\mathrm{E}}(x) =
\begin{cases}
0  & \text{if } x = 0, 1, 2, \\
f(x) & \text{if } x \in (0, 1), \\
-f(2-x) & \text{if } x \in (1, 2).
\end{cases}
$$

The source field values at are collocated at the grid points $x_k = \frac{2 k}{N}, \quad k = 0, 1, \dotsc, N-1$, and are collected into the antisymmetrically extended vector

$$
𝐟_{\mathrm{E}} =
\begin{pmatrix}
0 & f(x_1) & \dotsc & f(x_{N/2-1}) & 0 & -f(x_{N/2-1}) & \dotsc & -f(x_1)
\end{pmatrix}^{\top}.
$$

We introduce the antisymmetric extension matrix $𝐑 \in ℝ^{N \times N/2}$ such that

$$
𝐟_{\mathrm{E}} = 𝐑~ 𝐟,
$$

where $𝐟 \in ℝ^{N/2}$ contains grid point values only associated with the physical domain. The DFT of the source vector $𝐟_{\mathrm{E}}$ is defined as

$$
\hat{𝐟}_{\mathrm{E}} = 𝐅_N ~𝐟_{\mathrm{E}}.
$$

We can obtain the solution equation using

$$
𝐮 = 𝐑^{\top}𝐅_N^{\dagger}𝐑 ~ 𝐃^{-1} ~ 𝐑^{\top} 𝐅_N 𝐑 ~ 𝐟,
$$

where

$$
𝐃 =
\left(\frac{\pi}{L}\right)^2 diag \bigl(1 &emsp; 1 &emsp; 2^2 &emsp; 3^2 &emsp; \dotsc &emsp; (N/2-1)^2 \bigr).
$$

---
## Quantum implementation

We choose first the discretisation for our 1D Poisson-Dirchlet problem over the domain $\Omega = (0, 1)$. In this case we choose $n_{pts} = 2^5$ corresponding to $n_q = 5$ qubits. 

        n_state_qubits     = 5                          # physical domain \Omega
        n_state_qubits_ext = n_state_qubits + 1         # extended domain \Omega_E
    
        n_pts              = 2**n_state_qubits
        n_pts_ext          = 2**n_state_qubits_ext

We then discretise the force vector 

        # discretize the domain interval (0, 1)
        h     = 1/n_pts 
        x     = np.linspace(0, 1-h, n_pts)

        # forcing vector as input state
        f_i    = forcing(x, 1)                         # evaluate forcing function at collocation points
        f_i[0] = 0.                                    # make sure zero at the first element 
        norm_f = np.linalg.norm(f_i)                   # for normalisation
        f      = f_i/norm_f                            # normalised input state

Now we can setup our quantum circuit, starting with the qubit registers

        qcs = QuantumRegister(n_state_qubits, 'x')     # physical space 
        qce = QuantumRegister(1, 'e')                  # extension for antisymmetric reflection
        qca = QuantumRegister(n_ancillas, 'a')         # ancillary qubits
        qc  = QuantumCircuit(qcs, qce, qca)

The input state, which is the (normalised) discretised force vector 𝐟, can be prepared using a built-in Qiskit initialisation
        
        qc.initialize(f, qcs)

To obtain the final state $𝐮$ we require implementation of unitary operators for reflection, QFT, and polynomial encoding

        # reflection unitary
        qr = QuantumReflection(n_state_qubits, 1)
        qr_gate = qr.build()

        # qft
        qft = QFT(num_qubits=n_state_qubits_ext,inverse=False).to_gate()

        # piecewise polynomial approximation
        degree      = 3       
        breakpoints = [1, degree+1, 2*degree+1, 3*degree+1, 5*degree+1, 8*degree+1,  n_pts] 
        poly_coeffs = polynomials(twoTheta, degree, breakpoints)
        pw_approximation = PiecewisePolynomialPauliRotationsGate(n_state_qubits, breakpoints, poly_coeffs)

We can now append these unitary operators into the circuit following the classical algorithm

        qc.append(qr_gate, list(range(n_state_qubits_ext)))            # reflection
        qc.append(qft, list(range(n_state_qubits_ext)))                # QFT
        qc.append(qr_inv_gate, list(range(n_state_qubits_ext)))        # conj. transpose of reflection

        qc.append(pw_approximation, list(range(n_state_qubits+2)))     # solver in Fourier space

        qc.append(qr_gate, list(range(n_state_qubits_ext)))            
        qc.append(qft.inverse(), list(range(n_state_qubits_ext))) 
        qc.append(qr_inv_gate, list(range(n_state_qubits_ext))) 

The quantum state evolution through the unitaries can be performed as

        state = Statevector.from_label('0' * qc.num_qubits)
        state = state.evolve(qc)

Extract the result from the state and re-normalise to obtain the quantum spectral approximation of the solution 𝐮

        out    = np.array(state.data[0:n_pts])
        result = np.real(norm_f*out)

---

## Citation

If you use this repository in academic work, please cite:

    @article{Febrianto2025QSM,
      title   = {A Quantum Spectral Method for non-periodic boundary value problems},
      author  = {Eky Febrianto, Yiren Wang, Burigede Liu, Michael Ortiz, Fehmi Cirak},
      journal = {arXiv preprint arXiv:2511.11494},
      year    = {2025}
    }

---

