"""
Module for space time projection based model reduction using two PODs.
Reference: arXiv:2102.03505v1
"""
import numpy as np
from .reduction import Reduction
from .pod import POD

class SpaceTimePOD(Reduction):
    def __init__(self, method='tailored', optimal_modal_coefficients=True,
        spatial_pod_args={}, temporal_pod_args={}):
        """
        """
        available_methods = {
            'tailored': self._tailored_spacetime_basis,
            'kronecker': self._kronecker_spacetime_basis,
            'nested': self._nested_spacetime_reduction,
        }

        method = available_methods.get(method)
        if method is None:
            raise RuntimeError(
                "Invalid method for POD. Please chose one among {}".format(
                    ', '.join(available_methods)))
        self._method = method

        self._modes = None
        self._singular_values = None
        self._optimal_modal_coefficients = optimal_modal_coefficients

        self._spatial_pod_args = spatial_pod_args
        self._temporal_pod_args = temporal_pod_args

        self._modes_product_inv = None
        self._modes_inv = None

    @property
    def modes(self):
        return self._modes

    # methods

    def _kronecker_spacetime_basis(self, X1, X2):
        spatial_pod = POD(**self._spatial_pod_args)
        spatial_pod.reduce(X1)
        Phi = spatial_pod.modes

        temporal_pod = POD(**self._temporal_pod_args)
        temporal_pod.reduce(X2)
        Psi = temporal_pod.modes

        self._modes = np.kron(Psi, Phi)

    def _tailored_spacetime_basis(self, X1, X2):
        spatial_pod = POD(**self._spatial_pod_args)
        spatial_pod.reduce(X1)
        Phi = spatial_pod.modes

        self._temporal_pod = POD(**self._temporal_pod_args)
        self._modes = np.hstack([self._tailored_basis_piece(spatial_mode=u,
            X2=X2) for u in Phi.T])

    def _tailored_basis_piece(self, spatial_mode, X2):
        X2Phi_i = np.hstack((X2[:, i * spatial_mode.shape[0] : (i + 1) *
            spatial_mode.shape[0]] @ spatial_mode)[:, None]
            for i in range(self._ntrain))
        self._temporal_pod.reduce(X2Phi_i)
        tailored_Psi_i = self._temporal_pod.modes
        return np.kron(tailored_Psi_i, spatial_mode[:,None])

    def _nested_spacetime_reduction(self, X1):
        self._spatial_pod = POD(**self._spatial_pod_args)
        self._spatial_modal_coeffs = self._spatial_pod.reduce(X1)

        coeffstime_mu = np.reshape(self._spatial_modal_coeffs, (-1, self._ntrain),
            'F')
        self._temporal_pod = POD(**self._temporal_pod_args)
        return self._temporal_pod.reduce(coeffstime_mu)

    # POD interface

    def reduce(self, X):
        """
        Reduces the parameter Space by using the specified reduction method (default svd).

        :type: numpy.ndarray
        """
        # X axes:
        # - 0 -> time
        # - 1 -> space
        # - 2 ->  parameters

        # temp axes:
        # - 0 -> space
        # - 1 -> time
        # - 2 ->  parameters
        temp = np.swapaxes(X, 0, 1)

        # we store these values to use later when reconstructing the snapshots
        # in reduce()
        self._space_points = temp.shape[0]
        self._time_instants = temp.shape[1]
        self._ntrain = temp.shape[2]

        X1 = np.reshape(temp, (temp.shape[0], -1), 'F')

        if (self._method == self._tailored_spacetime_basis or
            self._method == self._kronecker_spacetime_basis):
            X2 = np.reshape(X, (X.shape[0], -1), 'F')

            if self._method == self._tailored_spacetime_basis:
                self._tailored_spacetime_basis(X1, X2)
            else:
                self._kronecker_spacetime_basis(X1, X2)

            # rows: space/time
            # columns: parameters
            X3 = np.reshape(temp, (-1, temp.shape[2]), 'F')

            if self._optimal_modal_coefficients:
                if self._modes_product_inv is None:
                    # initialize the constant matrix
                    self._modes_product_inv = (np.linalg.inv(self.modes.T
                        @ self.modes) @ self.modes.T)
                # compute modal coefficients
                return self._modes_product_inv @ X3
            else:
                if self._modes_inv is None:
                    # initialize the constant matrix
                    self._modes_inv = np.linalg.pinv(self.modes)
                # compute modal coefficients
                return self._modes_inv @ X3
        else:
            return self._nested_spacetime_reduction(X1)

    def expand(self, X):
        """
        Projects a reduced to full order solution.

        :type: numpy.ndarray
        """
        if (self._method == self._tailored_spacetime_basis or
            self._method == self._kronecker_spacetime_basis):
            expanded = self.modes.dot(X)

            space_time = np.reshape(expanded, (self._space_points,
                    self._time_instants, -1), 'F')
            # users expect time/space, not space/time
            return np.swapaxes(space_time, 0, 1)
        else:
            coeffstime_mu = self._temporal_pod.expand(X).T
            coeffs_timemu = np.reshape(coeffstime_mu,
                (-1, self._time_instants * self._ntrain), 'F')

            X1 = self._spatial_pod.expand(coeffs_timemu).T
            return np.swapaxes(np.reshape(X1,
                (self._space_points, self._time_instants, self._ntrain), 'F'),
                0, 1)
