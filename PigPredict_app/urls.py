from django.contrib import admin
from django.urls import path
from . import views

app_name = 'PigPredict_app'

urlpatterms = [
    path('forms/', views.get_forms, name='forms'),
    path('training/', views.training, name='training'),

]