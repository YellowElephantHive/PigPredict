import sys
import os
import numpy as np
import pandas as pd
import random


def distance_to(target, weight=None):
    if type(target) is not np.ndarray:
        target = np.array(target, dtype=np.float32)
    target = target.astype(np.float32)

    def distance_from(source):
        if weight:
            return np.array(np.sqrt((((source - target) ** 2) * weight).sum(axis=1)))
        else:
            return np.linalg.norm(source - target, axis=1)

    return distance_from


# TODO improvement: 根據是否為艷色使用不同的 distance 算法
class DyeSelector:

    def __init__(self, delta_limit=None, history_num_limit=1, max_collection_count=10):
        # params for _update_dye_collections_from_history
        self.delta_limit = delta_limit
        self.history_num_limit = history_num_limit
        self._dye_collections_from_history = None
        self._dye_collections_from_history_rows = None

        # params for _update_similar_collections()
        self._similar_dye_collections = None

        # params for get_possible_collections()
        self.max_collections_count = max_collection_count

    def get_possible_collections(self, df_all, df_single, target_labch, con_base=1.5, clothe='FVF2429'):
        self._update_dye_collections_from_history(df_all, target_labch)
        self._update_similar_collections(df_single, con_base, clothe)
        sample_size = len(self._similar_dye_collections)
        result = [*self._dye_collections_from_history, *random.sample(self._similar_dye_collections, k=sample_size)]
        return result[:self.max_collections_count]

    def _update_dye_collections_from_history(self, df_all, target_labch):
        """
        從歷史資料中，以 self.labch 靠近的資料使用的染料作為此次要使用的染料。
        小於 self.delta_limit 的資料都會被挑選出來。
        the method will mutate instance property _dye_collections_from_history
        """
        # 取得有染料濃度的資料
        df_tmp = df_all[np.logical_not(df_all['LAB'].str.contains('-0'))].copy()
        df_tmp = df_tmp.loc[df_tmp['染料_1'].notnull(),]
        distance_to_target = distance_to(target_labch, [1, 1, 1, 1, 1.5])
        sorted_idx = np.argsort(distance_to_target(df_tmp[['L', 'a', 'b', 'C', 'H']]))
        df_sorted = df_tmp.iloc[sorted_idx, :]

        # 取得距離小於 delta_limit 的資料
        if self.delta_limit:
            target_rows = df_sorted.iloc[distance_to_target(df_sorted[['L', 'a', 'b', 'C', 'H']]) < self.delta_limit, :]
        else:
            target_rows = df_sorted.iloc[:self.history_num_limit, :]
        self._dye_collections_from_history_rows = target_rows
        self._dye_collections_from_history = _get_dye_labels(target_rows)

    def _update_similar_collections(self, df_single, con_base=1.5, clothe='FVF2429'):
        """
        根據 self._dye_collections_from_history 的染料組合，從 df_single 中挑出相近的染料，並回傳相近似染料組合。
        the method will mutate instance property _similar_dye_collections
        """
        # 找單色相近染料時，使用濃度1.5，布料為FVF2429的資料
        df_single = df_single[(df_single['濃度_1'] == con_base) & (df_single['布號'] == clothe)]

        t = Traverser()

        # collection is a set {dye1, dye2...}
        for collection in self._dye_collections_from_history:
            # candidates_collection is a list containing sets
            # [{dye1, dye1-1...}...]
            candidates_collection = self._get_candidate_collection(df_single, collection)
            t.traverse(0, candidates_collection)

        similar_collections = t.get_collections
        similar_collections = similar_collections.difference(set(self._dye_collections_from_history))
        self._similar_dye_collections = similar_collections

    def _get_candidate_collection(self, df_single, collection):
        candidates_collection = []

        for i in range(len(collection)):
            candidates_collection.append(set())

        for idx, label in enumerate(collection):
            candidates_collection[idx].add(label)
            target_labch = df_single[(df_single['染料_1'] == label)]
            if len(target_labch) == 0:
                continue
            target_labch = target_labch.iloc[0, :][['L', 'a', 'b', 'C', 'H']].values
            distance_to_target = distance_to(target_labch, [1, 1, 1, 1, 1.5])
            sorted_idx = np.argsort(distance_to_target(df_single.loc[:, ['L', 'a', 'b', 'C', 'H']]))
            df_sorted = df_single.iloc[sorted_idx, :]
            candidates_collection[idx].update(df_sorted.iloc[1:3]['染料_1'].values)

        return candidates_collection


def _get_dye_labels(target_rows):
    """
    取得染料編號
    :param target_rows: a dataframe has columns l a b c h 染料_{1234}
    :return: return a python list contains tuple [(dye1, dye2...)...]
    """
    result = []
    for index, row in target_rows.iterrows():
        labels = [row[f'染料_{i}'] for i in range(1, 5) if row[f'染料_{i}'] is not np.nan]
        labels = frozenset(labels)
        result.append(labels)

    # remove duplicate and still remain order in the list
    result = np.array(result)
    _, idx = np.unique(result, return_index=True)
    result = result[np.sort(idx)]
    result = [tuple(sorted(i)) for i in result]

    return result


class Traverser():

    def __init__(self):
        self._tmp = []
        self._collections = set()

    def traverse(self, step, l):
        if step >= len(l):
            self._collections.add(tuple(self._tmp))
            return
        # traverse all iterables in l
        for i in range(len(l)):
            # 下一步是從下一個 iterable 裡面挑的
            if i == len(self._tmp):
                # traverse all dyes in the iterable
                for dye in l[i]:
                    self._tmp.append(dye)
                    self.traverse(len(self._tmp), l)
                    self._tmp.pop()

    @property
    def get_collections(self):
        return self._collections
