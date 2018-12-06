from django import forms
from .models import LAB_Clothe




class LAB_and_Clothe_Form(forms.ModelForm):
    class Meta:
        model = LAB_Clothe
        fields = '__all__'
