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
        X2 = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]), 'F')
        temp = np.swapaxes(X, 0, 1)
        X1 = np.reshape(temp, (temp.shape[0], temp.shape[1] * temp.shape[2]), 'F')

        if self._tailored:
            self._tailored_spacetime_basis(X1, X2)
        else:
            self._standard_spacetime_basis(X1, X2)

        return self.modes.T.dot(X)

    def _standard_spacetime_basis(self, X1, X2):
        spatial_pod = POD(**self._spatial_pod_args)
        spatial_pod.reduce(X1)
        Phi = spatial_pod.modes

        temporal_pod = POD(**self._temporal_pod_args)
        temporal_pod.reduce(X2)
        Psi = temporal_pod.modes

        self._modes = np.kron(Phi, Psi)

    def _tailored_spacetime_basis(self, X1, X2):
        spatial_pod = POD(**self._spatial_pod_args)
        spatial_pod.reduce(X1)
        Phi = spatial_pod.modes

        ntrain = int(X1.shape[1] / X2.shape[0])

        temporal_pod = POD(**self._temporal_pod_args)
        self._modes = np.block([self._tailored_basis_piece(spatial_mode=u, X2=X2,
            temporal_pod=temporal_pod, ntrain=ntrain) for u in Phi.T])

    def _tailored_basis_piece(self, spatial_mode, X2, temporal_pod, ntrain):
        XPhi_i = np.block([X2[:, i * spatial_mode.shape[0]] @ spatial_mode
            for i in range(ntrain)])
        temporal_pod.reduce(XPhi_i)
        tailored_Psi_i = temporal_pod.modes
        return np.kron(spatial_mode, tailored_Psi_i)

    def expand(self, X):
        """
        Projects a reduced to full order solution.

        :type: numpy.ndarray
        """
        return self.modes.dot(X).T
