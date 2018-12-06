import os
import sys
import pickle
import unittest
import collections

prj_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(prj_root)
from util.pickle_util import _PickleManager
from util.pickle_util import pm as pm_


class PickleUtilTestCase(unittest.TestCase):

    def test_pickle_manager(self):
        pm = _PickleManager()

        # test singleton
        self.assertIs(pm, pm_)

        # test pickle
        self.assertIsInstance(pm.decoder_input_columns, collections.Iterable)
        self.assertGreaterEqual(pm.lab_mean[0], 0)
        self.assertGreaterEqual(pm.lab_std[0], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
