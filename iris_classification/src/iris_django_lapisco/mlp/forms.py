from django import forms
from .models import MLP

class MLPForms(forms.ModelForm):
    class Meta:
        model = MLP
        
        fields = ['valores', 'classe']