"""
Module for space time projection based model reduction using two PODs.
Reference: arXiv:2102.03505v1
"""
import numpy as np
from .reduction import Reduction
from .pod import POD

class SpaceTimePOD(Reduction):
    def __init__(self, tailored=True, spatial_pod_args={}, temporal_pod_args={}):
        """
        """
        self._tailored = tailored
        self._spatial_pod_args = spatial_pod_args
        self._temporal_pod_args = temporal_pod_args

        self._modes_product_inv = None

    @property
    def modes(self):
        return self._modes

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

        # we store this values to use later when reconstructing the snapshots
        # in reduce()
        self._space_points = temp.shape[0]
        self._time_instants = temp.shape[1]
        self._ntrain = temp.shape[2]

        X1 = np.reshape(temp, (temp.shape[0], temp.shape[1] * temp.shape[2]), 'F')
        X2 = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]), 'F')

        if self._tailored:
            self._tailored_spacetime_basis(X1, X2)
        else:
            self._standard_spacetime_basis(X1, X2)

        # rows: space/time
        # columns: parameters
        X3 = np.reshape(temp, (temp.shape[0] * temp.shape[1], temp.shape[2]), 'F')

        if self._modes_product_inv is None:
            self._modes_product_inv = (np.linalg.inv(self.modes.T @ self.modes)
                @ self.modes.T)
        return self._modes_product_inv @ X3

    def _standard_spacetime_basis(self, X1, X2):
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

    def expand(self, X):
        """
        Projects a reduced to full order solution.

        :type: numpy.ndarray
        """
        expanded = self.modes.dot(X)

        if X.ndim == 2:
            # X contains modal coefficients for two or more parameters
            space_time = np.reshape(expanded, (self._space_points,
                self._time_instants, X.shape[1]), 'F')
        else:
            space_time = np.reshape(expanded, (self._space_points,
                self._time_instants), 'F')
        return np.swapaxes(space_time, 0,1)
