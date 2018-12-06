import tensorflow as tf
import numpy as np
import pickle
import sys
import os
import tempfile

import keras

prj_root = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(prj_root)
from util.normalize import *
from util.pickle_util import pm

# pickle objs back
decoder_input_columns = pm.decoder_input_columns
vec_dim = len(decoder_input_columns)


class InverseDecoder:
    def __init__(self, decoder_path, monte_carlo_count=200, epoch=2000, early_stop_loss=1e-5, lr=1e-2):
        self.tf_ref = {}
        self.decoder_path = decoder_path
        self.monte_carlo_count = monte_carlo_count
        self.early_stop_loss = early_stop_loss
        self.epoch = epoch
        self._lr = lr
        self._sess = None
        self._save_decoder_weight()

    def build_graph(self):
        """
        在 tensor flow 的 default graph 創造我們要用來找反函數的 graph。
        """
        tf.reset_default_graph()
        keras.backend.clear_session()

        with tf.name_scope('input'):
            self.tf_ref['inputs'] = tf.placeholder(dtype=tf.float32, shape=(self.monte_carlo_count, vec_dim))
            self.tf_ref['lab_target'] = tf.placeholder(dtype=tf.float32, shape=(self.monte_carlo_count, 3))

        with tf.variable_scope('dummy_var'):
            # tensor flow 不能直接對 place holder 做 gradient descent，所以先將 input 的值 assign 給 w
            # 如此就可以讓 tensor flow 幫我們做 gradient descent
            w = tf.get_variable('dummy_var', shape=(self.monte_carlo_count, vec_dim), dtype=tf.float32)
            a1 = tf.assign(w, self.tf_ref['inputs'])

        with tf.variable_scope('decoder'):
            with tf.variable_scope('model'):
                decoder = keras.models.load_model(os.path.join(prj_root, self.decoder_path))
                with tf.control_dependencies([a1]):
                    x = w
                    for layer in decoder.layers:
                        x = layer(x)
                    self.tf_ref['decoder_logits'] = x

        with tf.variable_scope('loss'):
            self.tf_ref['loss'] = tf.losses.mean_squared_error(self.tf_ref['lab_target'], self.tf_ref['decoder_logits'])

        with tf.variable_scope('opt'):
            opt = tf.train.AdamOptimizer(self._lr)
            with tf.control_dependencies([a1]):
                update = opt.minimize(self.tf_ref['loss'], var_list=[w])

        with tf.name_scope('output'):
            with tf.control_dependencies([update]):
                self.tf_ref['con_output'] = tf.identity(w)

        # saving weight, will be used in other methods
        self.tf_ref['decoder_weights'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder/model')
        self.tf_ref['opt_weights'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='opt')

    def init_model(self, sess):
        self._sess = sess
        sess.run(tf.global_variables_initializer())
        tf.train.Saver(self.tf_ref['decoder_weights']).restore(sess, self.tf_ref['weights_path'])

    def _init_opt(self):
        inits = [var.initializer for var in self.tf_ref['opt_weights']]
        self._sess.run(inits)

    def _save_decoder_weight(self):
        g = tf.Graph()
        with tf.Session(graph=g) as sess:
            with tf.variable_scope('decoder'):
                with tf.variable_scope('model'):
                    keras.models.load_model(os.path.join(prj_root, self.decoder_path))

            weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder/model')

            with tempfile.NamedTemporaryFile() as f:
                self.tf_ref['weights_path'] = tf.train.Saver(weights).save(sess, f.name)

    def predict_lab(self, concentrations):
        model = keras.models.load_model(self.decoder_path)
        y_pred = model.predict(concentrations)
        y_pred = unnormalize_lab(y_pred)
        return y_pred

    def lab_loss(self, concentrations, lab_target: 'a list'):
        lab = self.predict_lab(concentrations)
        lab_target = np.array(lab_target)
        return np.sqrt(((lab - lab_target) ** 2).sum(axis=1))

    def predict_concentrations(self, dyes_collections, clothe_type, lab_target: 'a list'):
        return np.array([self.predict_concentration(dyes,
                                                    clothe_type,
                                                    lab_target)
                         for dyes in dyes_collections])

    def predict_concentration(self, dyes, clothe_type, lab_target: 'a list'):
        self._init_opt()

        concentrations = self._get_init_concentrations(dyes, clothe_type)

        lab_target = np.array([lab_target] * self.monte_carlo_count)
        lab_target = normalize_lab(lab_target)

        print(dyes)

        for i in range(1, self.epoch + 1):
            fd = {self.tf_ref['inputs']: concentrations, self.tf_ref['lab_target']: lab_target}
            new_concentrations, loss_ = self._sess.run([self.tf_ref['con_output'], self.tf_ref['loss']], fd)
            concentrations[concentrations > 0] = new_concentrations[concentrations > 0]
            self._set_clothe_type(concentrations, clothe_type)
            if loss_ < self.early_stop_loss:
                break
            if i % 100 == 0:
                print(f'epoch_{i}, loss:{loss_:.8f}')

        return concentrations.mean(axis=0)

    def _get_init_concentrations(self, dyes, clothe_type):

        result = np.zeros((self.monte_carlo_count, vec_dim), dtype=np.float32)

        dye_idx = [np.where(decoder_input_columns == f'concentration_{dye}')[0][0] for dye in dyes]

        # initialize range from 0.5 to 2.5
        result[:, dye_idx] = np.random.rand(self.monte_carlo_count, len(dye_idx)) * 2. + 0.5

        self._set_clothe_type(result, clothe_type)

        return result

    def _set_clothe_type(self, concentration, clothe_type):
        clothe_idx = np.where(decoder_input_columns == clothe_type)[0][0]
        concentration[:, clothe_idx] = 1
