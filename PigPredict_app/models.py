from django.db import models

# Create your models here.



class LAB_Clothe(models.Model):
    L = models.FloatField()
    A = models.FloatField()
    B = models.FloatField()
    CLOTHE = models.CharField(max_length=256)

