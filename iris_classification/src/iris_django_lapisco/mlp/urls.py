from django.urls import path, include

from .views import mlp,resultado, adicionar_amostra, atualizar_consulta, deletar_consulta, classificar

urlpatterns = [
    
    path('', mlp, name='mlp'),
    path('resultado/<str:valores>/', resultado),
    path('add', adicionar_amostra, name='adicionar_amostra'),
    path('update/<int:id>/', atualizar_consulta, name='atualizar_consulta'),
    path('delete/<int:id>/', deletar_consulta, name='deletar_consulta'),
    path('classificar/<int:id>/', classificar, name='classificar')
]