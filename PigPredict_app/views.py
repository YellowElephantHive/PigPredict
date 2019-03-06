import os
import sys
import time

from django.shortcuts import render, HttpResponseRedirect
from skimage import color

from .forms import LAB_and_Clothe_Form
from .models import LAB_Clothe

# np.random.seed(88)
# random.seed = 88

prj_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(prj_root)
from .core.modify_pred import modify_pred
from .core.dyeselector import *
from .core.inverse_decoder import *
from .util.pickle_util import pm
from .util.config_util import cm
from .util.dfutil import *

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


# Create your views here.

def index(request):
    return render(request, 'PigPredict_app/index_2.html')

def get_forms(request):
    form = LAB_and_Clothe_Form()

    if request.method == 'POST':
        form = LAB_and_Clothe_Form(request.POST)
        if form.is_valid():
            form.save(commit=True)
            return HttpResponseRedirect(request.path_info)

        else:
            print("Please type correct data")

    return render(request, 'PigPredict_app/base.html', {'form':form})

def training(request):
    LAB_Info = LAB_Clothe.objects.values_list()
    l = LAB_Info[LAB_Info.count()-1][1]
    a = LAB_Info[LAB_Info.count()-1][2]
    b = LAB_Info[LAB_Info.count()-1][3]
    clothe = LAB_Info[LAB_Info.count()-1][4]


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

    df_output = df_output[['布號', '染料_1', '濃度_1', '染料_2', '濃度_2', '染料_3', '濃度_3', '染料_4', '濃度_4', 'delta_lab']]

    context = {'output_table': df_output}

    return render(request, 'PigPredict_app/index.html', context)

#def BackToFormPage(request):

