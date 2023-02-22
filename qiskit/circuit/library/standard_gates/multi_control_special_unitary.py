# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""ControlledGate classes for multicontrolled unitary gates"""
import numpy
from cmath import isclose
from typing import Union, List
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ControlledGate, Qubit
from qiskit.circuit._utils import check_su2
from .x import McxVchainDirty, LinearMcx
from .h import HGate


class Ldmcsu(ControlledGate):
    """
    Linear depth Multi-Controlled Gate for Special Unitary
    ------------------------------------------------

    Multicontrolled gate decomposition with linear cost.
    `unitary` must be a SU(2) matrix. The details of this 
    decomposition are presented in https://arxiv.org/abs/2302.06377
    """

    def __init__(
        self,
        unitary,
        num_controls,
        ctrl_state: str=None
    ):

        check_su2(unitary)
        self.unitary = unitary
        self.controls = QuantumRegister(num_controls)
        self.target = QuantumRegister(1)
        self.num_controls = num_controls + 1
        self.ctrl_state = ctrl_state

        super().__init__("ldmcsu", self.num_controls, [], "ldmcsu")

    def _define(self):

        self.definition = QuantumCircuit(self.controls, self.target)

        is_main_diag_real = isclose(self.unitary[0, 0].imag, 0.0) and \
                            isclose(self.unitary[1, 1].imag, 0.0)
        is_secondary_diag_real = isclose(self.unitary[0,1].imag, 0.0) and \
                                  isclose(self.unitary[1,0].imag, 0.0)

        if  not is_main_diag_real and not is_secondary_diag_real:
            # U = V D V^-1, where the entries of the diagonal D are the eigenvalues
            # `eig_vals` of U and the column vectors of V are the eigenvectors
            # `eig_vecs` of U. These columns are orthonormal and the main diagonal
            # of V is real-valued.
            eig_vals, eig_vecs = numpy.linalg.eig(self.unitary)

            x_vecs, z_vecs = self._get_x_z(eig_vecs)
            x_vals, z_vals = self._get_x_z(numpy.diag(eig_vals))

            self.half_linear_depth_mcv(
                x_vecs, z_vecs, self.controls, self.target, self.ctrl_state, inverse=True
            )
            self.linear_depth_mcv(
                x_vals,
                z_vals,
                self.controls,
                self.target,
                self.ctrl_state,
                general_su2_optimization=True
            )
            self.half_linear_depth_mcv(
                x_vecs, z_vecs, self.controls, self.target, self.ctrl_state
            )

        else:
            x, z = self._get_x_z(self.unitary)

            if not is_secondary_diag_real:
                self.definition.h(self.target)

            self.linear_depth_mcv(
                x,
                z,
                self.controls,
                self.target,
                self.ctrl_state
            )

            if not is_secondary_diag_real:
                self.definition.h(self.target)

    @staticmethod
    def _get_x_z(su2):
        is_secondary_diag_real = isclose(su2[0,1].imag, 0.0) and isclose(su2[1,0].imag, 0.0)

        if is_secondary_diag_real:
            x = su2[0,1]
            z = su2[1,1]
        else:
            x = -su2[0,1].real
            z = su2[1,1] - su2[0,1].imag * 1.0j

        return x, z

    def linear_depth_mcv(
        self,
        x,
        z,
        controls: Union[QuantumRegister, List[Qubit]],
        target: Qubit,
        ctrl_state: str=None,
        general_su2_optimization=False
    ):

        alpha_r = numpy.sqrt(
        (numpy.sqrt((z.real + 1.) / 2.) + 1.) / 2.
        )
        alpha_i = z.imag / (2. * numpy.sqrt(
            (z.real + 1.) * (numpy.sqrt((z.real + 1.) / 2.) + 1.)
        ))
        alpha = alpha_r + 1.j * alpha_i
        beta = x / (2. * numpy.sqrt(
                (z.real + 1.) * (numpy.sqrt((z.real + 1.) / 2.) + 1.)
            )
        )

        s_op = numpy.array(
            [[alpha, -numpy.conj(beta)],
            [beta, numpy.conj(alpha)]]
        )

        # S gate definition
        s_gate = QuantumCircuit(1)
        s_gate.unitary(s_op, 0)

        num_ctrl = len(controls)
        k_1 = int(numpy.ceil(num_ctrl / 2.))
        k_2 = int(numpy.floor(num_ctrl / 2.))

        ctrl_state_k_1 = None
        ctrl_state_k_2 = None

        if ctrl_state is not None:
            ctrl_state_k_1 = ctrl_state[::-1][:k_1][::-1]
            ctrl_state_k_2 = ctrl_state[::-1][k_1:][::-1]

        if not general_su2_optimization:
            mcx_1 = McxVchainDirty(k_1, ctrl_state=ctrl_state_k_1).definition
            self.definition.append(mcx_1, controls[:k_1] + controls[k_1:2*k_1 - 2] + [target])
        self.definition.append(s_gate, [target])

        mcx_2 = McxVchainDirty(
            k_2, ctrl_state=ctrl_state_k_2, action_only=general_su2_optimization
        ).definition
        self.definition.append(mcx_2.inverse(), controls[k_1:] + controls[k_1 - k_2 + 2:k_1] + [target])
        self.definition.append(s_gate.inverse(), [target])

        mcx_3 = McxVchainDirty(k_1, ctrl_state=ctrl_state_k_1).definition
        self.definition.append(mcx_3, controls[:k_1] + controls[k_1:2*k_1 - 2] + [target])
        self.definition.append(s_gate, [target])

        mcx_4 = McxVchainDirty(k_2, ctrl_state=ctrl_state_k_2).definition
        self.definition.append(mcx_4, controls[k_1:] + controls[k_1 - k_2 + 2:k_1] + [target])
        self.definition.append(s_gate.inverse(), [target])

    def half_linear_depth_mcv(
        self,
        x,
        z,
        controls: Union[QuantumRegister, List[Qubit]],
        target: Qubit,
        ctrl_state: str=None,
        inverse: bool=False
    ):

        alpha_r = numpy.sqrt((z.real + 1.) / 2.)
        alpha_i = z.imag / numpy.sqrt(2*(z.real + 1.))
        alpha = alpha_r + 1.j * alpha_i

        beta = x / numpy.sqrt(2*(z.real + 1.))

        s_op = numpy.array(
            [[alpha, -numpy.conj(beta)],
            [beta, numpy.conj(alpha)]]
        )

        # S gate definition
        s_gate = QuantumCircuit(1)
        s_gate.unitary(s_op, 0)

        # Hadamard equivalent definition
        h_gate = QuantumCircuit(1)
        h_gate.unitary(numpy.array([[-1, 1], [1, 1]]) * 1/numpy.sqrt(2), 0)

        num_ctrl = len(controls)
        k_1 = int(numpy.ceil(num_ctrl / 2.))
        k_2 = int(numpy.floor(num_ctrl / 2.))

        ctrl_state_k_1 = None
        ctrl_state_k_2 = None

        if ctrl_state is not None:
            ctrl_state_k_1 = ctrl_state[::-1][:k_1][::-1]
            ctrl_state_k_2 = ctrl_state[::-1][k_1:][::-1]

        if inverse:
            #self.definition.h(target)
            self.definition.append(HGate(), [target], [])

            self.definition.append(s_gate, [target])
            mcx_2 = McxVchainDirty(k_2, ctrl_state=ctrl_state_k_2, action_only=True).definition
            self.definition.append(mcx_2, controls[k_1:] + controls[k_1 - k_2 + 2:k_1] + [target])

            self.definition.append(s_gate.inverse(), [target])

            self.definition.append(h_gate, [target])

        else:
            mcx_1 = McxVchainDirty(k_1, ctrl_state=ctrl_state_k_1).definition
            self.definition.append(mcx_1, controls[:k_1] + controls[k_1:2*k_1 - 2] + [target])
            self.definition.append(h_gate, [target])

            self.definition.append(s_gate, [target])

            mcx_2 = McxVchainDirty(k_2, ctrl_state=ctrl_state_k_2).definition
            self.definition.append(mcx_2, controls[k_1:] + controls[k_1 - k_2 + 2:k_1] + [target])
            self.definition.append(s_gate.inverse(), [target])

            self.definition.append(HGate(), [target], [])
