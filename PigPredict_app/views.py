from typing import Optional, Any

from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.http import HttpRequest, HttpResponse
from django.views.generic import (View, TemplateView,
                                ListView, DetailView,
                                CreateView, DeleteView,
                                UpdateView)
from . import models
from .forms import LAB_and_Clothe_Form
from .models import LAB_Clothe
from django.views.decorators.csrf import csrf_exempt


import numpy as np
import os
import sys
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
from .preprocess import dyeprocess
from .core.modify_pred import modify_pred
from .core.validator import validate
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
'''
def index(request):
    return render(request, 'PigPredict_app/index.html')
'''

'''
def base(request):

    if request.method == 'POST':

        l = request.POST.get('l', 100)
        a = request.POST.get('a', 128)
        b = request.POST.get('b', 128)
        clothe = request.POST.get('option', "FVF2666")

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

        return HttpRequest(request, 'PigPredict_app/base.html')

'''

'''
@csrf_exempt
def base(request):
    post_list = LAB_Clothe.objects.order_by('id')
    form = LAB_and_Clothe_Form()
    return render(request, 'PigPredict_app/base.html',
                  {'post': post_list,
                   'form': form},)



def base_2(request):
    if request.method == 'POST':
        form = LAB_and_Clothe_Form(request.POST)
        if form.is_valid():
            form.save(commit=False)
    return render(request, 'PigPredict_app/base.html',
                  {'post': post_list,
                   'form': form},)

@csrf_exempt
def training(request):
    data = {
        "app_name": "PigPredict_app",
        "random_number": random.randint(0, 10000)
    }

    if request.method == 'GET':
        form = LAB_and_Clothe_Form()
        data.update({"form": form, "submit": True})

    elif request.method == 'POST':
        form = LAB_and_Clothe_Form(request.POST)
        l = request.POST.get('l', 100)
        a = request.POST.get('a', 128)
        b = request.POST.get('b', 128)
        clothe = request.POST.get('option', "FVF2666")
        if request.POST.get('submit'):
            LAB_Clothe_model = LAB_Cl(
                L=l,
                A=a,
                B=b,
                CLOTHE=clothe
            )

            LAB_Clothe_model.save()


    return HttpResponse("fuck you")

'''

def index(request):
    return render(request, 'PigPredict_app/base.html')

def form(request):
    form = LAB_and_Clothe_Form()

    if request.method == 'POST':
        form = LAB_and_Clothe_Form(request.POST)
        if form.is_valid():
            form.save(commit=True)
            return index(request)

        else:
            print("fuck you")

    return render(request, 'PigPredict_app/base.html', {'form':form})

# def training(request):
