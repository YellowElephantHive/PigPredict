import os
import sys
import pickle
import unittest
import collections

prj_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(prj_root)

from util.config_util import cm


class ConfigUtilTestCase(unittest.TestCase):

    def test_config_manager(self):
        self.assertTrue(cm['general']['DecoderInputColumnsPath'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
