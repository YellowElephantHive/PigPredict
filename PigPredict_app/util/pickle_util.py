import configparser
import os
import sys
import pickle

prj_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(prj_root)
from util.config_util import cm


class Singleton(object):
    _instance = None

    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kw)
        return cls._instance


class _PickleManager(Singleton):

    def __init__(self):
        decoder_input_columns_path = os.path.join(prj_root, cm['general']['DecoderInputColumnsPath'])
        lab_mean_path = os.path.join(prj_root, cm['general']['LabMeanPath'])
        lab_std_path = os.path.join(prj_root, cm['general']['LabStdPath'])

        with open(decoder_input_columns_path, 'rb') as f:
            self.decoder_input_columns = pickle.load(f)
        with open(lab_mean_path, 'rb') as f:
            self.lab_mean = pickle.load(f)
        with open(lab_std_path, 'rb') as f:
            self.lab_std = pickle.load(f)


pm = _PickleManager()
