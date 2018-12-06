# coding: utf-8

import sys
import os
import numpy as np
import pandas as pd
import random
import datetime
import time


def val_l(l):
    if l >= 0 and l <= 100:
        return True
    else:
        print("-l:請輸入0~100的值")
        return False


def val_a(a):
    if a >= -128 and a <= 128:
        return True
    else:
        print("-a:請輸入-128~128的值")
        return False


def val_b(b):
    if b >= -128 and b <= 128:
        return True
    else:
        print("-b:請輸入-128~128的值")
        return False


def val_clothe_type(clothe_type):
    if clothe_type == 'FVF2184' or clothe_type == 'FVF2429' or clothe_type == 'FVF2666':
        return True
    else:
        print("-clothe:請輸入以下三種布料(FVF2184, FVF2429, FVF2666)")
        return False


def validate(_l, _a, _b, _clothe):
    try:
        l = float(_l)
        if val_l(l) == False:
            sys.exit(0)
    except ValueError:
        print("-l:請輸入0~100的值")
        sys.exit(0)

    try:
        a = float(_a)
        if val_a(a) == False:
            sys.exit(0)
    except ValueError:
        print("-a:請輸入-128~128的值")
        sys.exit(0)

    try:
        b = float(_b)
        if val_b(b) == False:
            sys.exit(0)
    except ValueError:
        print("-b:請輸入-128~128的值")
        sys.exit(0)

    try:
        clothe_type = _clothe
        if val_clothe_type(clothe_type) == False:
            sys.exit(0)
    except ValueError:
        print("-clothe:請輸入以三種布料(FVF2184, FVF2429, FVF2666)")
    return l, a, b, clothe_type
