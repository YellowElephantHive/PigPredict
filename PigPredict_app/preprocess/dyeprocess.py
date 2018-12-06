import numpy as np
import pandas as pd
import re
import collections
from collections import OrderedDict
from functools import lru_cache

DYE_DF_ALL_COL_NAMES = [f'染料_{i}' for i in range(1, 5)]
DYE_DF_SINGLE_COL_NAMES = [f'染料_{i}' for i in range(1, 2)]
MAX_DYE_NUM_LEN = 4


def get_total_dye_names(df_all, df_single) -> set:
    """
    :param df_all:
    :param df_single:
    :return: 回傳所有在 df_all 和 df_single 的染料代號
    """
    dye_names_all = get_unique_values_in_cols(df_all, DYE_DF_ALL_COL_NAMES)
    dye_names_single = get_unique_values_in_cols(df_single, DYE_DF_SINGLE_COL_NAMES)
    total_dye_names = list(dye_names_all.union(dye_names_single))
    total_dye_names.sort()
    return total_dye_names


def get_unique_values_in_cols(df, col_names) -> set:
    if isinstance(col_names, str):
        dye_names = {name for name in np.unique(df.loc[df[col_names].notnull(), [col_names]])}

    elif isinstance(col_names, collections.Iterable):
        dye_names = {name for col in col_names
                     for name in np.unique(df.loc[df[col].notnull(), [col]])}

    return dye_names


def get_total_count(dye_name, df_all, df_single):
    """
    :param dye_name:
    :param df_all:
    :param df_single:
    :return: 回傳 dye_name 染料在 df_all 和 df_single 出現的次數
    """
    count = 0
    count += get_count(dye_name, df_all)
    count += get_count(dye_name, df_single)
    return count


def get_count(dye_name, df):
    """
    :param dye_name:
    :param df:
    :return: 回傳 dye_name 染料 在 df 內出現的次數
    """
    count = 0
    for i in range(1, 1 + MAX_DYE_NUM_LEN):
        try:
            count += (df[f'染料_{i}'].astype(str) == dye_name).sum()
        except KeyError:
            break
    return count


def df_dye_one_hot_with_concetration(df, df_all, df_single, min_occur=0):
    """
    :param df:
    :param df_all:
    :param df_single:
    :param min_occur:
    :return: 以 df_all 和 df_single 內至少出現 min_occur 次數的染料做 one hot encoding
             然後 在 df 增加 one hot encoding 的欄位，欄位名稱為 'concentration_{one hot col name}
             (ex: concentration_264-1)
             然後把每一列染料濃度填入對應的欄位
    """
    dye_name_to_one_hot_vec, aborted_dye = dye_name_to_one_hot_mapping(df_all, df_single, min_occur)

    df_drop = drop_rows_by_dyes(df, aborted_dye)

    concentration_vector = cal_concentration_vecs(df_drop, dye_name_to_one_hot_vec)

    for key in dye_name_to_one_hot_vec:
        df_drop[f'concentration_{key}'] = 0
    dye_columns = [f'concentration_{k}' for k in dye_name_to_one_hot_vec.keys()]
    df_drop[dye_columns] = concentration_vector

    return df_drop


def dye_name_to_one_hot_mapping(df_all, df_single, min_occur=0):
    """
    :param df_all:
    :param df_single:
    :param min_occur:
    :return:
    dye_name_to_one_hot_dim_order: 染料名稱對應的 one hot vector {染料名稱: [0,..., 1, 0,...]} (不包含出現次數<min_occur的染料)
    aborted_dye: 出現次數不滿 min_occur 的染料們
    """
    total_dye_names = get_total_dye_names(df_all, df_single)
    dye_names_count_mapping = {name: get_total_count(name, df_all, df_single) for name in total_dye_names}
    dye_names_count_mapping = {k: v for k, v in dye_names_count_mapping.items() if v >= min_occur}

    dye_name_to_one_hot_vec = OrderedDict()
    eye = np.eye(len(dye_names_count_mapping))
    for idx, key in enumerate(sorted(dye_names_count_mapping)):
        dye_name_to_one_hot_vec[key] = eye[idx]

    aborted_dye = [k for k, v in dye_names_count_mapping.items() if v < min_occur]

    return dye_name_to_one_hot_vec, aborted_dye


def drop_rows_by_dyes(df, aborted_dye):
    """
    如果該列出有任一 aborted_dye 染料，則 drop 該列
    :param df:
    :param aborted_dye:
    :return:
    """
    abort_condition = False
    for i in range(1, 1 + MAX_DYE_NUM_LEN):
        try:
            abort_condition = np.logical_or(abort_condition, np.in1d(df[f'染料_{i}'].values, aborted_dye))
        except KeyError:
            break

    df_drop = df.loc[np.logical_not(abort_condition), :].copy().reset_index(drop=True)
    return df_drop


def cal_concentration_vecs(df, dye_name_to_one_hot):
    """
    取得濃度向量  ex [0, 0,..., 0.5, 0,..., 1.36, 0, ...]
    :param df:
    :param dye_name_to_one_hot:
    :return: [[濃度向量1],
              [濃度向量2]...]
    """
    result = 0

    for i in range(1, 1 + MAX_DYE_NUM_LEN):

        try:
            one_hot_dye = np.array(
                [dye_name_to_one_hot.get(key, np.zeros(len(dye_name_to_one_hot))) for key in df[f'染料_{i}']])
            concentration = df[f'濃度_{i}'].copy()
            concentration[concentration.isnull()] = 0

            for j in range(len(concentration)):
                one_hot_dye[j, :] = one_hot_dye[j, :] * float(concentration[j])

            result += one_hot_dye

        except KeyError:
            break

    return result
