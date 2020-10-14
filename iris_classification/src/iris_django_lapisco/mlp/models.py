from django.db import models

# Create your models here.
class MLP(models.Model):
    valores = models.CharField(max_length=100)
    classe = models.CharField(max_length=100)

    def __str__(self):
        return self.valores