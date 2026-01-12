import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate, U3Gate

class QuantumReflection:
    """
    Build the quantum circuit implementing the operator U_R.

    Args:
        n_q (int): Number of data qubits.
        n_b (int): Number of boundary/auxiliary qubits (used in defining T).
        method_u_f (str): Forward shift method. Supported:
            - "mcx": multi-controlled 
            - "ripple-carry": Ripple-carry utilising ancillas

    Attributes:
        n_q (int): Number of data qubits.
        n_b (int): Number of boundary/auxiliary qubits.
        n_t (int): Total number of logical qubits (data + boundary).
        n_dst (int): Total number of qubits used in the circuit (includes ancillas if needed).
        method_u_f (str): Method for forward shift unitary
        qct (QuantumCircuit): The constructed circuit named "T".
    """

    def __init__(self, n_q, n_b, method_u_f="mcx"):
        # Define parameters
        self.n_q = n_q
        self.n_b = n_b
        self.n_t =  n_q + n_b
        self.method_u_f = method_u_f

        # Determine circuit width
        if method_u_f == "mcx":
            self.n_dst = self.n_t
        elif method_u_f == "ripple-carry":
            self.n_dst = (2* n_q - 1) + n_b

        self.qct = QuantumCircuit(self.n_dst, name="U_R")
        
        # Add the U_B unitary: U_R_0 . U_R_1 . U_R_2
        self._add_u_b()
        
        # Add the forward shift unitary (U_R_3)
        self._add_forward_shift_u_r3()
        
    def _add_u_b(self):
        """
        Append the Unitary U_B = U_R_0 . U_R_1 . U_R_3 to the circuit.
        """
        # unitary U_R_0
        self.qct.x(self.n_q)
        gateB = U3Gate(np.pi / 2, 0., 3. * np.pi / 2, label='B')
        self.qct.append(gateB, [self.n_q])
        
        # unitary U_R_1
        self.qct.append(gateB.inverse().control(self.n_q, ctrl_state=0), list(range(self.n_t)))
        
        # unitary U_R_2 using CX ladder
        for j in range(self.n_q):
            self.qct.cx(self.n_q, self.n_q - j - 1)
            
    def _add_forward_shift_u_r3(self):
        """
        Append the forward shift unitary U_R_3 operation to the circuit.
        The unitary U_R_3 is U_F with one control qubit

        Returns:
            Appends controlled U_R_3 to `self.qct`.
        """
        if self.method_u_f == "mcx":
            qcp = QuantumCircuit(self.n_q, name="U_F")
            for i in range(self.n_q-1)[::-1]:
                ctrl = range(i+1)
                mcx = XGate().control(num_ctrl_qubits=i+1, ctrl_state='1'*(i+1))
                qcp.append(mcx, list(ctrl) + [i+1])
            qcp.x(0)
            self.qct.append(qcp.control(1, ctrl_state=1), [self.n_q] + list(range(self.n_q)))

        elif self.method_u_f == "ripple-carry":
            qcp = QuantumCircuit(2 * self.n_q - 1, name="U_F")
            
            # Alter the first two qubits
            qcp.x(0)
            qcp.cx(0, 1, ctrl_state=0)
            qcp.cx(0, self.n_q, ctrl_state=0)
            
            for j in range(1, self.n_q - 1):
                qcp.ccx(j, self.n_q + j - 1, self.n_q + j, ctrl_state='10')
                qcp.cx(self.n_q + j, j + 1, ctrl_state=1)
                
            # Undo ancilla
            for j in range(1, self.n_q - 1):
                qcp.ccx(
                    self.n_q - 1 - j, 
                    (2 * (self.n_q - 1)) - j, 
                    self.n_q + self.n_q - 1 - j, 
                    ctrl_state='10',
                )
            qcp.cx(0, self.n_q, ctrl_state='0')
            
            # Convert to controlled gate
            self.qct.append(
                qcp.control(1, ctrl_state=1), 
                list(np.roll(range(self.n_t), 1)) + list(range(self.n_t, 2 * self.n_q)),
            )

    ### Public API
    def build(self):
        """
        Return the constructed U_R circuit.
        """
        return self.qct
    
    def inverse(self):
        """
        Return the inverse circuit of U_R.
        """
        return self.qct.inverse()

