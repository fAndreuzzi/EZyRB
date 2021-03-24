"""
Module for the snapshots database collected during the Offline stage
"""
import numpy as np


class SpaceTimeDatabase(object):
    """
    Database class

    :param array_like parameters: the input parameters
    :param array_like time_instants: the time instants
    :param numpy.ndarray snapshots: the input snapshots, space varies along the
        first axis, time along the second, and parameters along the third axis.
    :param Scale scaler_parameters: the scaler for the parameters. Default
        is None meaning no scaling.
    :param Scale scaler_snapshots: the scaler for the snapshots. Default is
        None meaning no scaling.
    """
    def __init__(self,
                 parameters=None,
                 time_instants=None,
                 snapshots=None,
                 scaler_parameters=None,
                 scaler_snapshots=None):
        self._time_instants = None
        self._parameters = None
        self._snapshots = None
        self.scaler_parameters = scaler_parameters
        self.scaler_snapshots = scaler_snapshots

        # if only parameters or snapshots are provided

        provided = 0
        if parameters is not None:
            provided += 1
        if time_instants is not None:
            provided += 1
        if snapshots is not None:
            provided += 1

        if provided != 0 and provided != 3:
            raise RuntimeError('Wrong number of arguments')

        if parameters is not None and snapshots is not None:
            self.add(parameters, time_instants, snapshots)

    @property
    def parameters(self):
        """
        The matrix containing the input parameters (by row).

        :rtype: numpy.ndarray
        """
        if self.scaler_parameters:
            return self.scaler_parameters.fit_transform(self._parameters)
        else:
            return self._parameters

    @property
    def time_instants(self):
        """
        The array containing the time instants.

        :rtype: numpy.ndarray
        """
        return self._time_instants

    @property
    def snapshots(self):
        """
        The matrix containing the snapshots (by row).

        :rtype: numpy.ndarray
        """
        if self.scaler_snapshots:
            return self.scaler_snapshots.fit_transform(self._snapshots)
        else:
            return self._snapshots

    def __getitem__(self, indexes):
        """
        This method returns a new Database with the selected parameters and snapshots.

        .. warning:: The new parameters and snapshots are a view of the
            original Database.
        """
        # in this case the user provided only one argument
        if not isinstance(indexes, tuple):
            return self[indexes, :]

        parameters_index, time_index = indexes

        return SpaceTimeDatabase(parameters=self._parameters[parameters_index],
            time_instants=self._time_instants[time_index],
            snapshots=self._snapshots[parameters_index, :, time_index],
            scaler_parameters=self.scaler_parameters,
            scaler_snapshots=self.scaler_snapshots)

    def __len__(self):
        """
        This method returns the number of snapshots.

        :rtype: int
        """
        return self._snapshots.shape[1] * self._snapshots.shape[2]

    def add(self, parameters, time_instants, snapshots):
        """
        Add (by row) new sets of snapshots and parameters to the original
        database.

        :param array_like parameters: the parameters to add.
        :param array_like time_instants: the time instants to add.
        :param numpy.ndarray snapshots: the snapshots to add.
        """
        # check dimensions and type
        if not isinstance(snapshots, np.ndarray) or snapshots.ndim != 3:
            raise ValueError('The provided snapshot must be a 3D nparray')

        if (snapshots.shape[0] != len(parameters) or
            snapshots.shape[2] != len(time_instants)):
            raise ValueError("""Dimensions do not agree.\nsnapshots: {} time
                instants for {} parameters, but received {} time instants
                and {} parameters""" .format(snapshots.shape[2],
                snapshots.shape[0], len(time_instants), len(parameters)))

        if (self._parameters is None and self._time_instants is None and
            self._snapshots is None):
            self._parameters = parameters
            self._time_instants = time_instants
            self._snapshots = snapshots
        else:
            self._parameters = np.vstack([self._parameters, parameters])
            self._snapshots = np.vstack([self._snapshots, snapshots])

        return self
