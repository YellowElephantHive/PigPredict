import pickle
import os
from util.pickle_util import pm

prj_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)


def unnormalize_lab(lab: 'np array has 3 columns'):
    lab = lab.copy()
    lab_mean = pm.lab_mean
    lab_std = pm.lab_std
    lab[:, 0] = lab[:, 0] * lab_std['L'] + lab_mean['L']
    lab[:, 1] = lab[:, 1] * lab_std['a'] + lab_mean['a']
    lab[:, 2] = lab[:, 2] * lab_std['b'] + lab_mean['b']
    return lab


def normalize_lab(lab: 'np array has 3 columns'):
    lab = lab.copy()
    lab_mean = pm.lab_mean
    lab_std = pm.lab_std
    lab[:, 0] = (lab[:, 0] - lab_mean['L']) / lab_std['L']
    lab[:, 1] = (lab[:, 1] - lab_mean['a']) / lab_std['a']
    lab[:, 2] = (lab[:, 2] - lab_mean['b']) / lab_std['b']
    return lab
