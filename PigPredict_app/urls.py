from django.contrib import admin
from django.urls import path
from . import views

app_name = 'PigPredict_app'

urlpatterms = [
    path('', views.index, name='index'),
    path('base', views.form, name='form'),

]