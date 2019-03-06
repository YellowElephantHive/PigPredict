from django.test import TestCase
import django
import os
import unittest
from .models import LAB_Clothe
from .views import *
from faker import Faker
import random


# Create your tests here.

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'PigPredict.settings')
django.setup()

fake_data_gen = Faker()

def data_to_db(N=5):

    for row in range(N):

        fake_L = fake_data_gen.random.number()

