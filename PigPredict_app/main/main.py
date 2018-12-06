import sys
import os
import numpy as np
import pandas as pd
import random
import datetime
import time
from argparse import ArgumentParser
from skimage import color

# np.random.seed(88)
# random.seed = 88

prj_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(prj_root)
import preprocess.dyeprocess as dyeprocess
from core.modify_pred import modify_pred
from core.validator import validate
from core.dyeselector import *
from core.inverse_decoder import *
from util.pickle_util import pm
from util.config_util import cm
from util.dfutil import *

# get pickle objs back
decoder_input_columns = pm.decoder_input_columns

""" global vars """
DATA_FILE_ALL = os.path.join(prj_root, cm['general']['DataFileAll'])
DATA_FILE_SINGLE = os.path.join(prj_root, cm['general']['DataFileSingle'])
DECODER_PATH = os.path.join(prj_root, cm['general']['DecoderPath'])
OUTPUT_DIR = os.path.join(prj_root, cm['general']['OutputDir'])
con_dim = len(decoder_input_columns)
clothe_type_count = int(cm['general']['ClotheTypeCount'])

# for dye selector
history_num_limit = int(cm['dye_selector']['HistoryNumLimit'])
max_collection_count = int(cm['dye_selector']['MaxCollectionCount'])

# argparser
parser = ArgumentParser()
parser.add_argument("-l", "--l", dest="l", help="-l: 請輸入0~100的值")
parser.add_argument("-a", "--a", dest="a", help="-a: 請輸入-128~128的值")
parser.add_argument("-b", "--b", dest="b", help="-b: 請輸入-128~128的值")
parser.add_argument("-clothe", "--clothe", dest="clothe", help="-clothe: 請輸入以下三種布料(FVF2184, FVF2429, FVF2666)")

args = parser.parse_args()
l, a, b, clothe = validate(format(args.l), format(args.a), format(args.b), format(args.clothe))
# l, a, b, clothe = 29.64, -9.27, -17.33, 'FVF2666'

lch = color.lab2lch([float(l), float(a), float(b)])
c = lch[1]
h = lch[2] * 180 / np.pi
lab = [l, a, b]
labch = [l, a, b, c, h]

# loading data for selecting dyes
df_all = pd.read_csv(DATA_FILE_ALL)
df_all = df_all[df_all['abort'] != 1]
df_all = df_all[df_all['L'].notnull()]

df_single = pd.read_csv(DATA_FILE_SINGLE)
df_single = df_single[df_single['abort'] != 1]


""" 選擇染料組合 """
dye_selector = DyeSelector(history_num_limit=history_num_limit,
                           max_collection_count=max_collection_count)
possible_collections = dye_selector.get_possible_collections(df_all, df_single, labch)

""" 計算染料濃度 """
inverse_decoder = InverseDecoder(DECODER_PATH, 100)
inverse_decoder.build_graph()

with tf.Session() as sess:
    inverse_decoder.init_model(sess)
    pred = inverse_decoder.predict_concentrations(possible_collections, clothe, lab)
    pred = modify_pred(pred)
    losses = inverse_decoder.lab_loss(pred, [lab] * len(pred))

""" output result to csv """
df_output = process_output_csv(pred, losses, decoder_input_columns, clothe, clothe_type_count)
output_path = os.path.join(OUTPUT_DIR, f'{time.time()}.csv')
df_output.to_csv(output_path, index=False)
