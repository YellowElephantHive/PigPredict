# Generated by Django 2.1.2 on 2018-12-05 06:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('PigPredict_app', '0003_auto_20181130_1451'),
    ]

    operations = [
        migrations.AlterField(
            model_name='lab_clothe',
            name='CLOTHE',
            field=models.CharField(max_length=256),
        ),
    ]
