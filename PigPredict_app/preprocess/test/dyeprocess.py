import unittest
import pandas as pd
import sys
import os

from preprocess.dyeprocess import *

ALL_FILE = os.path.join(os.path.dirname(__file__), 'resource', 'sample_all.csv')
SINGLE_FILE = os.path.join(os.path.dirname(__file__), 'resource', 'sample_single.csv')


class DyeProcessTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_all = pd.read_csv(ALL_FILE)
        cls.df_single = pd.read_csv(SINGLE_FILE)

    def test_get_unique_values_in_cols(self):
        dye_names_all = get_unique_values_in_cols(DyeProcessTestCase.df_all, ['染料_1', '染料_2', '染料_3', '染料_4'])
        dye_names_single = get_unique_values_in_cols(DyeProcessTestCase.df_single, ['染料_1'])

        actual_names_all = {'322-6', '249-6', '461-9', '432', 'F4G-6', 'F6G-6', 'F6S', '227-1', '244-4', '264-1'}
        actual_names_single = {'264-1', 'F6S', '261-8'}

        self.assertSetEqual(dye_names_all, actual_names_all)
        self.assertSetEqual(dye_names_single, actual_names_single)

    def test_get_total_dye_names(self):
        total_dye_names = get_total_dye_names(DyeProcessTestCase.df_all, DyeProcessTestCase.df_single)
        actual_names = {'322-6',
                        '249-6',
                        '461-9',
                        '432',
                        'F4G-6',
                        'F6G-6',
                        'F6S',
                        '227-1',
                        '244-4',
                        '264-1',
                        '261-8'}

        self.assertSetEqual(set(total_dye_names), actual_names)

    def test_get_count(self):
        count = get_count('F6S', DyeProcessTestCase.df_all)
        self.assertEqual(count, 2)

    def test_get_total_count(self):
        count = get_total_count('F6S', DyeProcessTestCase.df_all, DyeProcessTestCase.df_single)
        self.assertEqual(count, 6)

    def test_df_dye_one_hot_with_concetration(self):
        df = df_dye_one_hot_with_concetration(DyeProcessTestCase.df_all, DyeProcessTestCase.df_all,
                                              DyeProcessTestCase.df_single)
        self.assertAlmostEqual(df.loc[1, 'concentration_322-6'], 0.327, places=5)
        self.assertAlmostEqual(df.loc[5, 'concentration_F4G-6'], 0.363, places=5)

        df = df_dye_one_hot_with_concetration(DyeProcessTestCase.df_single, DyeProcessTestCase.df_all,
                                              DyeProcessTestCase.df_single)
        self.assertAlmostEqual(df.loc[0, 'concentration_264-1'], 0.06, places=5)

        df = df_dye_one_hot_with_concetration(DyeProcessTestCase.df_all, DyeProcessTestCase.df_all,
                                              DyeProcessTestCase.df_single, min_occur=4)

        self.assertAlmostEqual(df.loc[df['LAB'] == 'F17BNF004-26', 'concentration_F6S'].values, 1.33, places=5)

if __name__ == '__main__':
    unittest.main(verbosity=2)
