from django.contrib import admin
from django.urls import path
from PigPredict_app import views

urlpatterns = [
    path("", views.index, name="index"),
    path("forms/", views.legacy_entry, name="forms"),
    path("training/", views.legacy_entry, name="training"),
    path("api/predict/", views.predict, name="predict"),
    path("api/health/", views.health, name="health"),
    path("admin/", admin.site.urls),
]
