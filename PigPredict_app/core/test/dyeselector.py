import unittest
import pandas as pd
import sys
import os

from core.dyeselector import *

prj_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

ALL_FILE = os.path.join(os.path.dirname(__file__), 'resource', 'hong_make_all_revised_2.csv')
SINGLE_FILE = os.path.join(os.path.dirname(__file__), 'resource', 'hong_make_single_revised.csv')


class DyeProcessTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_all = pd.read_csv(ALL_FILE)
        cls.df_single = pd.read_csv(SINGLE_FILE)

    def test_get_dye_collections_from_history(self):
        df_all = pd.read_csv(ALL_FILE)
        df_all = df_all[df_all['abort'] != 1]
        df_all = df_all[np.logical_not(df_all['LAB'].str.contains('-\s*0\s*$', case=True, regex=True))]

        df_single = pd.read_csv(SINGLE_FILE)
        df_single = df_single[df_single['abort'] != 1]

        target_labch = df_all[['L', 'a', 'b', 'C', 'H']].values[0]
        dye_selector = DyeSelector(10)

        dye_selector._update_dye_collections_from_history(df_all, target_labch)
        dye_collections = dye_selector._dye_collections_from_history
        self.assertIn('322-6', dye_collections[0])
        self.assertIn('249-6', dye_collections[0])
        self.assertIn('461-9', dye_collections[0])

    def test_get_similar_collections(self):
        df_all = pd.read_csv(ALL_FILE)
        df_all = df_all[df_all['abort'] != 1]
        df_all = df_all[np.logical_not(df_all['LAB'].str.contains('-\s*0\s*$', case=True, regex=True))]

        df_single = pd.read_csv(SINGLE_FILE)
        df_single = df_single[df_single['abort'] != 1]

        target_labch = df_all[['L', 'a', 'b', 'C', 'H']].values[0]
        dye_selector = DyeSelector()

        dye_selector._update_dye_collections_from_history(df_all, target_labch)
        dye_selector._update_similar_collections(df_single)
        dye_collections = dye_selector._similar_dye_collections
        print(dye_collections)
        print(len(dye_collections))
        # 249-6 沒有出現在 single 內 所以 candidate collection 為 [{'227-1', '322-6', '327'}, {'264-1', 'F6S', '461-9'}, {'249-6'}]
        # 共 3*3 九種可能，扣掉我們選出來的那一組，所以是8個
        self.assertTrue(len(dye_collections) == 8)

    def test_get_possible_collections(self):
        df_all = pd.read_csv(ALL_FILE)
        df_all = df_all[df_all['abort'] != 1]
        df_all = df_all[np.logical_not(df_all['LAB'].str.contains('-\s*0\s*$', case=True, regex=True))]

        df_single = pd.read_csv(SINGLE_FILE)
        df_single = df_single[df_single['abort'] != 1]

        target_labch = df_all[['L', 'a', 'b', 'C', 'H']].values[0]
        dye_selector = DyeSelector()

        dye_collections = dye_selector.get_possible_collections(df_all, df_single, target_labch)
        print(len(dye_collections))

        self.assertTrue(len(dye_collections) == 9)
        self.assertIn('322-6', dye_collections[0])
        self.assertIn('249-6', dye_collections[0])
        self.assertIn('461-9', dye_collections[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
