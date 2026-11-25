from django.urls import path
from . import views

app_name = "PigPredict_app"
urlpatterns = [
    path("", views.index, name="index"),
    path("forms/", views.legacy_entry, name="forms"),
    path("training/", views.legacy_entry, name="training"),
]
