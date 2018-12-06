import configparser
import os
import sys

prj_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(prj_root)
setting_path = os.path.join(prj_root, 'setting.cfg')


class _ConfigManager():

    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.config = configparser.ConfigParser()
        self.config.read(self.cfg_path)

    def load(self, cfg_path=None):
        if cfg_path:
            self.cfg_path = cfg_path
        self.config.read(self.cfg_path)

    def __getitem__(self, item):
        return self.config[item]


cm = _ConfigManager(setting_path)
