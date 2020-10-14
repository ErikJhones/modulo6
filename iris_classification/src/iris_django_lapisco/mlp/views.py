from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpRequest
from os import path
from decouple import config as cfg
from django.utils.datastructures import MultiValueDictKeyError
from django.views.decorators.http import require_http_methods
from .models import MLP
from .forms import MLPForms

from .utils import load_models, check_inputs, convert_string_to_list

#load models
model, tf = load_models()

# Create your views here.
def nome(request):
    nome = 'La'
    return render(request, 'mlp.html', {'nome':nome})

def mlp(request):
    resultado = 'Resultado classificação'
    result = MLP.objects.all()
    #digita o valor da lista para classificar no browser
    print(result[0].valores)
    index = len(result)-1
    for i in range(len(result)):
        lista_amostra = convert_string_to_list(result[i].valores)
        x = check_inputs(lista_amostra)
        y_hat = model.predict(tf.transform(x))
        result[i].classe = y_hat

    #depois usar o classificador e chamar a resposta
    print("printar a resposta no browser")
    return render(request, "mlp.html", {'result': result})

def resultado(request, valores):
    resultados = MLP.objects.get(id=valores)
    return render(request, "mlp.html", {'resultados': resultados})

def adicionar_amostra(request):
    form = MLPForms(request.POST or None, initial={'classe': 'default'})

    if form.is_valid():
        form.save()   
        return redirect('mlp')

    return render(request, 'mlp-form.html', {'form':form})

def atualizar_consulta(request, id):
    valor = MLP.objects.get(id=id)
    print(valor)
    form = MLPForms(request.POST or None, instance=valor)

    if form.is_valid():
        form.save()
        return redirect('mlp')

    return render(request, 'mlp-form.html', {'form': form, 'valor': valor})

def deletar_consulta(request, id):
    valor = MLP.objects.get(id=id)

    if request.method == 'POST':
        valor.delete()
        return redirect('mlp')
    
    return render(request, 'mlp-delete-confirm.html', {'valor': valor})

def classificar(request, id):
    valor = MLP.objects.get(id=id)

    print(valor.valores)
    lista_amostra = convert_string_to_list(valor.valores)

    x = check_inputs(lista_amostra)
    y_hat = model.predict(tf.transform(x))
    return render(request, 'mlp.html', {'y_hat': y_hat})