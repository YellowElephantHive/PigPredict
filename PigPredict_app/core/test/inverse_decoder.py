import tensorflow as tf
import unittest
import pickle
import sys
import os

prj_root = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(prj_root)
from core.inverse_decoder import *
from util.normalize import *

lab_mean_path = os.path.join(prj_root, 'resource', 'pickled', 'lab_mean.pkl')
lab_std_path = os.path.join(prj_root, 'resource', 'pickled', 'lab_std.pkl')
concentration_path = os.path.join(prj_root, 'resource', 'pickled', 'decoder_x_val.pkl')
lab_path = os.path.join(prj_root, 'resource', 'pickled', 'decoder_y_val.pkl')
decoder_input_columns_path = os.path.join(prj_root, 'resource', 'pickled', 'decoder_input_columns.pkl')

with open(lab_mean_path, 'rb') as f:
    lab_mean = pickle.load(f)
with open(lab_std_path, 'rb') as f:
    lab_std = pickle.load(f)
with open(concentration_path, 'rb') as f:
    concentration = pickle.load(f)
with open(lab_path, 'rb') as f:
    lab = pickle.load(f)
with open(decoder_input_columns_path, 'rb') as f:
    decoder_input_columns = pickle.load(f)

decoder_path = os.path.join(prj_root, 'resource', 'keras_model', 'keras_decoder_model')
clothe_type_count = 3


class InverseDecoderTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.inverse_model = InverseDecoder(decoder_path, 2000)

    def test_load_weights(self):
        inverse_model = InverseDecoder(os.path.join(prj_root, 'resource', 'keras_model', 'keras_decoder_model'), 1)

        inverse_model.build_graph()
        sess = tf.Session()
        inverse_model.init_model(sess)

        t = concentration.iloc[0:1, :]
        true_lab = lab.iloc[0, :].copy().values.reshape(1, 3)
        true_lab = unnormalize_lab(true_lab)

        fd = {inverse_model.tf_ref['inputs']: t}

        decoder_logits_ = sess.run(inverse_model.tf_ref['decoder_logits'], fd)
        decoder_logits_ = unnormalize_lab(decoder_logits_)

        print(decoder_logits_)
        print(true_lab)
        self.assertLess(((decoder_logits_ - true_lab) ** 2).sum(), 10)

        sess.close()

    def test_predict_lab(self):
        inverse_model = InverseDecoderTestCase.inverse_model

        t = concentration.iloc[0:1, :]
        true_lab = lab.iloc[0, :].copy().values.reshape(1, 3)
        true_lab = unnormalize_lab(true_lab)

        lab_pred = inverse_model.predict_lab(t)

        print(lab_pred)
        print(true_lab)
        self.assertLess(((lab_pred - true_lab) ** 2).sum(), 10)

    def test_get_init_concentrations(self):
        inverse_model = InverseDecoder(os.path.join(prj_root, 'resource', 'keras_model', 'keras_decoder_model'), 1)
        idx = np.where(concentration.iloc[0, :int(-1 * clothe_type_count)] > 0)[0]
        dyes = [column.split('_')[1] for column in decoder_input_columns[idx]]

        clothe_idx = len(decoder_input_columns) - clothe_type_count + \
                     np.where(concentration.iloc[0, int(-1 * clothe_type_count):] > 0)[0][0]
        clothe_type = decoder_input_columns[clothe_idx]

        concentrations = inverse_model._get_init_concentrations(dyes, clothe_type)

        print(concentrations)
        self.assertAlmostEqual((concentrations > 0).sum(), len(dyes) + 1)

    def test_predict_concentration(self):
        inverse_model = InverseDecoderTestCase.inverse_model

        inverse_model.build_graph()
        sess = tf.Session()
        inverse_model.init_model(sess)

        t = concentration.iloc[0:1, :]
        idx = np.where(concentration.iloc[0, :int(-1 * clothe_type_count)] > 0)[0]
        dyes = [column.split('_')[1] for column in decoder_input_columns[idx]]

        clothe_idx = len(decoder_input_columns) - clothe_type_count + \
                     np.where(concentration.iloc[0, int(-1 * clothe_type_count):] > 0)[0][0]
        clothe_type = decoder_input_columns[clothe_idx]

        true_lab = lab.iloc[0, :].copy().values.reshape(1, 3)
        true_lab = unnormalize_lab(true_lab)

        con = inverse_model.predict_concentration(dyes,
                                                  clothe_type,
                                                  true_lab[0].tolist())
        print(con)
        print(t.values)
        self.assertLess(np.abs(con - t.values).sum(), 1)
        sess.close()

    def test_predict_concentrations(self):
        inverse_model = InverseDecoderTestCase.inverse_model

        inverse_model.build_graph()
        sess = tf.Session()
        inverse_model.init_model(sess)

        t = concentration.iloc[0:1, :]
        idx = np.where(concentration.iloc[0, :int(-1 * clothe_type_count)] > 0)[0]
        dyes = [column.split('_')[1] for column in decoder_input_columns[idx]]

        clothe_idx = len(decoder_input_columns) - clothe_type_count + \
                     np.where(concentration.iloc[0, int(-1 * clothe_type_count):] > 0)[0][0]
        clothe_type = decoder_input_columns[clothe_idx]

        true_lab = lab.iloc[0, :].copy().values.reshape(1, 3)
        true_lab = unnormalize_lab(true_lab)

        con = inverse_model.predict_concentrations([dyes] * 2,
                                                   clothe_type,
                                                   true_lab[0].tolist())
        print(con)
        print(t.values)
        self.assertLess(np.abs(con - t.values).sum(), 1)

        sess.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
