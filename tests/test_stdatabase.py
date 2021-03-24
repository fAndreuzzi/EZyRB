import numpy as np

from unittest import TestCase
from ezyrb import STDatabase


class TestStDatabase(TestCase):
    def test_constructor_empty(self):
        a = STDatabase()

    def test_constructor_arg(self):
        db = STDatabase(np.random.uniform(size=(10, 3)),
                 np.random.uniform(size=(100)),
                 np.random.uniform(size=(10, 50, 100)))

        assert db.parameters.shape == (10, 3)
        assert db.time_instants.shape == (100,)
        assert db.snapshots.shape == (10,50,100)

    def test_constructor_arg_wrong(self):
        with self.assertRaises(ValueError):
            STDatabase(np.random.uniform(size=(10, 3)),
                 np.random.uniform(size=(100)),
                 np.random.uniform(size=(9, 50, 100)))

    def test_constructor_arg_wrong2(self):
        with self.assertRaises(ValueError):
            STDatabase(np.random.uniform(size=(10, 3)),
                 np.random.uniform(size=(100)),
                 np.random.uniform(size=(10, 50, 101)))

    def test_constructor_error(self):
        with self.assertRaises(RuntimeError):
            STDatabase(np.eye(5))

    def test_constructor_error2(self):
        with self.assertRaises(RuntimeError):
            STDatabase(np.eye(5), np.eye(5))

    def test_getitem(self):
        org = STDatabase(np.random.uniform(size=(10, 3)),
                 np.random.uniform(size=(100)),
                 np.random.uniform(size=(10, 50, 100)))
        new = org[::2, 90::3]

        assert new.parameters.shape[0] == 5
        assert len(new.time_instants) == 4
        assert new.snapshots.shape == (5, 50, 4)

    def test_getitem_one_argument(self):
        org = STDatabase(np.random.uniform(size=(10, 3)),
                 np.random.uniform(size=(100)),
                 np.random.uniform(size=(10, 50, 100)))
        new = org[::2]

        assert new.parameters.shape[0] == 5
        assert len(new.time_instants) == 100
        assert new.snapshots.shape == (5,50,100)
