
from django.contrib import admin
from django.urls import path, include
from django.contrib import admin
from django.conf.urls import url
from mlp import views,urls

urlpatterns = [
    path('', include('mlp.urls')),
    path('admin/', admin.site.urls),
    
]
